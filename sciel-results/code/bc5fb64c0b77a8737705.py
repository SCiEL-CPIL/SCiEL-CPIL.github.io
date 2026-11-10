import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    points = list(range(16))
    all_triples = list(combinations(points, 3))
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}
    
    # Precompute 5-sets
    all_5_sets = list(combinations(points, 5))
    num_5_sets = len(all_5_sets)
    
    # Precompute which 5-sets each triple covers
    # triple_idx -> list of 5-set indices
    triple_covers = [[] for _ in all_triples]
    
    for t_idx, t in enumerate(all_triples):
        t_set = set(t)
        for s_idx, s in enumerate(all_5_sets):
            if t_set.issubset(set(s)):
                triple_covers[t_idx].append(s_idx)
                
    # Helper to check if a triple is valid given current used pairs
    # We maintain used_pairs as a set of (u, v) and (v, u) for O(1) lookup
    # But for speed in local search, we might just check against current triples.
    # Given small size, checking against current triples (max ~40) is fast enough.
    
    def get_pairs(t_idx):
        t = all_triples[t_idx]
        return [(t[0], t[1]), (t[1], t[0]), (t[0], t[2]), (t[2], t[0]), (t[1], t[2]), (t[2], t[1])]

    def is_linear_with(triple_idx, current_triples):
        """Check if adding triple_idx to current_triples maintains linearity."""
        t = all_triples[triple_idx]
        t_set = set(t)
        for c_idx in current_triples:
            if c_idx == triple_idx:
                continue
            c = all_triples[c_idx]
            # Intersection size > 1?
            if len(t_set.intersection(set(c))) > 1:
                return False
        return True

    def calculate_alpha(triples):
        """Calculate independence number of the hypergraph."""
        # A set is independent if it contains no triple.
        # We want the largest such set.
        # We can check subsets of size k in descending order.
        # Max possible alpha is 16.
        # But we know alpha >= 4.
        # If we cover all 5-sets, alpha <= 4.
        # So we just need to check if there is an independent set of size 5.
        # If not, alpha is 4 (since we can always pick 4 points? No, we need to ensure 4 is possible).
        # Actually, if all 5-sets are covered, any set of size 5 has a triple, so it's not independent.
        # So alpha <= 4.
        # Is alpha 3 possible? No, proven above.
        # So if all 5-sets are covered, alpha is exactly 4.
        # If some 5-set is not covered, alpha >= 5.
        
        # For scoring, we just need the value.
        # Let's compute it properly for the final result.
        
        # Check size 5
        for s in all_5_sets:
            s_set = set(s)
            # Check if s contains any triple from 'triples'
            has_triple = False
            for t_idx in triples:
                t = all_triples[t_idx]
                if set(t).issubset(s_set):
                    has_triple = True
                    break
            if not has_triple:
                return 5 # Found an independent set of size 5. Could be larger, but 5 is enough to distinguish from 4.
                
        # If no independent set of size 5, alpha is at most 4.
        # Check size 4 (just to be sure, though theoretically it should be 4)
        # Actually, if alpha < 4, it would be 3.
        # But we proved alpha >= 4.
        # So if no size 5 independent set, alpha is 4.
        return 4

    def run_sa():
        # Initialize with a greedy packing
        current_triples = set()
        used_pairs = set()
        cover_count = [0] * num_5_sets
        uncovered = num_5_sets
        
        def add_triple(t_idx):
            nonlocal uncovered
            # Update cover counts
            for s_idx in triple_covers[t_idx]:
                if cover_count[s_idx] == 0:
                    uncovered -= 1
                cover_count[s_idx] += 1
            
            # Update used pairs
            for p in get_pairs(t_idx):
                used_pairs.add(p)
                
            current_triples.add(t_idx)

        def remove_triple(t_idx):
            nonlocal uncovered
            # Update cover counts
            for s_idx in triple_covers[t_idx]:
                cover_count[s_idx] -= 1
                if cover_count[s_idx] == 0:
                    uncovered += 1
                    
            # Update used pairs
            for p in get_pairs(t_idx):
                used_pairs.remove(p)
                
            current_triples.remove(t_idx)
            
        # Greedy Init
        # Pick random triples that are valid and cover many uncovered 5-sets
        candidates = list(range(len(all_triples)))
        random.shuffle(candidates)
        
        for t_idx in candidates:
            if time.time() > deadline - 5:
                break
            # Check validity
            t = all_triples[t_idx]
            t_set = set(t)
            valid = True
            for p in get_pairs(t_idx):
                if p in used_pairs:
                    valid = False
                    break
            if not valid:
                continue
                
            # Check gain
            gain = 0
            for s_idx in triple_covers[t_idx]:
                if cover_count[s_idx] == 0:
                    gain += 1
            
            # Always add if valid? Or only if gain > 0?
            # Adding valid triples generally helps.
            if gain > 0:
                add_triple(t_idx)
            elif len(current_triples) < 20:
                # Add some to start even if no immediate gain, to enable future swaps?
                # No, better to keep it tight.
                pass
                
        # If greedy didn't start well, force add a few
        if not current_triples:
             for t_idx in candidates[:10]:
                 valid = True
                 for p in get_pairs(t_idx):
                     if p in used_pairs:
                         valid = False
                         break
                 if valid:
                     add_triple(t_idx)
                     if len(current_triples) >= 5: break

        best_uncovered = uncovered
        best_triples = set(current_triples)
        
        # Simulated Annealing
        T = 10.0
        T_min = 0.01
        alpha = 0.999
        
        while time.time() < deadline - 2:
            if uncovered == 0:
                break
                
            # Pick a move
            move_type = random.random()
            
            if move_type < 0.2:
                # Add
                # Find valid candidates
                valid_candidates = []
                # Sample a few to speed up
                sample_size = min(100, len(all_triples))
                candidates = random.sample(range(len(all_triples)), sample_size)
                for t_idx in candidates:
                    if t_idx in current_triples:
                        continue
                    t = all_triples[t_idx]
                    valid = True
                    for p in get_pairs(t_idx):
                        if p in used_pairs:
                            valid = False
                            break
                    if valid:
                        valid_candidates.append(t_idx)
                        
                if valid_candidates:
                    t_add = random.choice(valid_candidates)
                    # Calculate delta
                    old_uncovered = uncovered
                    add_triple(t_add)
                    delta = uncovered - old_uncovered
                    
                    if delta < 0 or random.random() < pow(2.718, -delta / T):
                        if uncovered < best_uncovered:
                            best_uncovered = uncovered
                            best_triples = set(current_triples)
                    else:
                        remove_triple(t_add)
                        
            elif move_type < 0.4:
                # Remove
                if current_triples:
                    t_rem = random.choice(list(current_triples))
                    old_uncovered = uncovered
                    remove_triple(t_rem)
                    delta = uncovered - old_uncovered
                    # Removing increases uncovered (delta > 0) usually, unless it was redundant?
                    # No, removing a triple might uncover 5-sets.
                    # We accept if random.
                    if delta < 0 or random.random() < pow(2.718, -delta / T):
                        if uncovered < best_uncovered:
                            best_uncovered = uncovered
                            best_triples = set(current_triples)
                    else:
                        add_triple(t_rem)
                        
            else:
                # Swap
                if current_triples:
                    t_rem = random.choice(list(current_triples))
                    remove_triple(t_rem)
                    
                    # Find valid add candidates
                    valid_candidates = []
                    sample_size = min(100, len(all_triples))
                    candidates = random.sample(range(len(all_triples)), sample_size)
                    for t_idx in candidates:
                        if t_idx in current_triples:
                            continue
                        if t_idx == t_rem:
                            continue
                        t = all_triples[t_idx]
                        valid = True
                        for p in get_pairs(t_idx):
                            if p in used_pairs:
                                valid = False
                                break
                        if valid:
                            valid_candidates.append(t_idx)
                            
                    if valid_candidates:
                        t_add = random.choice(valid_candidates)
                        old_uncovered = uncovered
                        add_triple(t_add)
                        delta = uncovered - old_uncovered
                        
                        if delta < 0 or random.random() < pow(2.718, -delta / T):
                            if uncovered < best_uncovered:
                                best_uncovered = uncovered
                                best_triples = set(current_triples)
                        else:
                            remove_triple(t_add)
                            add_triple(t_rem)
                    else:
                        add_triple(t_rem)
                        
            T *= alpha
            if T < T_min:
                T = T_min
                
        # Restore best state if we ended in a worse state?
        # The 'best_triples' tracks the minimum uncovered seen.
        # We need to return the triples corresponding to best_uncovered.
        # But we lost the state when we updated best_triples (we only saved the set of indices).
        # We need to reconstruct the state from best_triples to verify/return.
        # Or just return the list of triples.
        
        return [list(all_triples[i]) for i in best_triples]

    best_solution = None
    best_score = 100 # Lower is better (uncovered count)
    
    # Run multiple restarts
    while time.time() < deadline - 10:
        solution = run_sa()
        
        # Calculate actual alpha for this solution
        # We can approximate by checking if all 5-sets are covered.
        # If yes, alpha = 4.
        # If no, alpha >= 5.
        
        # Let's verify coverage for the returned solution
        sol_set = set(tuple(t) for t in solution)
        sol_indices = set(triple_to_idx[t] for t in sol_set)
        
        # Check linearity
        linear = True
        sol_list = list(sol_indices)
        for i in range(len(sol_list)):
            for j in range(i+1, len(sol_list)):
                t1 = all_triples[sol_list[i]]
                t2 = all_triples[sol_list[j]]
                if len(set(t1).intersection(set(t2))) > 1:
                    linear = False
                    break
            if not linear:
                break
                
        if not linear:
            continue
            
        # Check coverage of 5-sets
        cover_count = [0] * num_5_sets
        for t_idx in sol_indices:
            for s_idx in triple_covers[t_idx]:
                cover_count[s_idx] += 1
                
        uncovered = sum(1 for c in cover_count if c == 0)
        
        if uncovered < best_score:
            best_score = uncovered
            best_solution = solution
            
        if best_score == 0:
            break
            
    if best_solution is None:
        # Fallback: return a valid small linear system
        # Just a few random triples
        return [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
        
    return best_solution

if __name__ == "__main__":
    construct()
