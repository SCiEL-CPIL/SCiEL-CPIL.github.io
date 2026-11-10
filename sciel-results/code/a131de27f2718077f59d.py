"""Linear triple systems on 16 points with small independence number."""
import random
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    random.seed(42)
    
    N = 16
    ALL_PAIRS = list(combinations(range(N), 2))
    PAIR_TO_IDX = {p: i for i, p in enumerate(ALL_PAIRS)}
    
    def is_linear(triples):
        """Check if any two triples share at most one point."""
        # A triple is linear with others if it doesn't share a pair with any other.
        # We can just check pairwise for small T, or use pair masks.
        # For speed in local search, we maintain pair masks.
        # Here we do a simple check for the final result or initial setup.
        for i in range(len(triples)):
            for j in range(i + 1, len(triples)):
                if len(set(triples[i]) & set(triples[j])) > 1:
                    return False
        return True

    def calculate_alpha(triples):
        """
        Calculate the independence number (largest subset with no triple).
        Uses branch and bound on the complement: 
        We want to find the max size of S such that no triple is fully contained in S.
        Equivalently, find the min size of C such that every triple intersects C (Hitting Set).
        alpha = N - min_hitting_set_size.
        
        However, checking "no triple in S" is easier to prune if we track which triples are "covered" (hit) by the complement?
        Actually, let's stick to finding Max Independent Set directly.
        A set S is independent if for all t in triples, S does not contain t.
        
        Branch and Bound on S:
        We try to include points. If we include a point, we must ensure we don't complete a triple.
        This is tricky.
        
        Alternative: Check all subsets is too slow if called too often.
        Let's use a specific B&B for Max Independent Set in Hypergraph.
        
        State: current_set_mask, remaining_candidates_mask, current_size.
        If current_size + popcount(remaining_candidates) <= best: prune.
        
        To prune effectively:
        If we add a point p to S, we must remove from 'remaining_candidates' any point q such that {p, q} is part of a triple that is "almost complete"?
        No, the constraint is: S cannot contain ANY triple.
        So if S contains {a, b}, it cannot contain c if {a, b, c} is a triple.
        
        Let's implement a simple B&B:
        1. Order points.
        2. Recursively decide to include or exclude point i.
        3. If include i: 
           - Check if adding i completes any triple with existing points in S. 
             If yes, invalid branch.
           - Also, for any triple containing i, if the other 2 points are in S, it's invalid.
           - If the other 1 point is in S, the 3rd point is FORBIDDEN from being added later.
        """
        # Precompute for each point, which triples contain it
        # And for each pair, which triple contains it (since it's linear, at most 1 triple per pair)
        pair_to_triple = {}
        for i, t in enumerate(triples):
            for p in combinations(t, 2):
                pair_to_triple[p] = i
        
        # point_to_triples
        point_to_triples = [[] for _ in range(N)]
        for i, t in enumerate(triples):
            for p in t:
                point_to_triples[p].append(i)
        
        best_alpha = 0
        
        # We will use a recursive search with bitmasks for speed
        # mask: points in current set S
        # forbidden_mask: points that cannot be added to S because they would complete a triple with points already in S
        # candidates_mask: points not yet decided
        
        # To speed up, we can pre-sort points or just iterate 0..15.
        # Let's use a simple iterative deepening or just standard B&B.
        
        def popcount(x):
            return bin(x).count('1')
            
        def search(current_mask, forbidden_mask, candidates_mask, current_size):
            nonlocal best_alpha
            
            if current_size + popcount(candidates_mask) <= best_alpha:
                return
            
            if candidates_mask == 0:
                if current_size > best_alpha:
                    best_alpha = current_size
                return
            
            # Pick a point from candidates
            # Heuristic: pick point with highest degree in remaining graph? 
            # For simplicity, pick lowest index bit.
            # Optimization: If a point is in forbidden_mask, it's not in candidates_mask.
            
            # Get lowest set bit in candidates_mask
            # p = (candidates_mask & -candidates_mask).bit_length() - 1
            p_bit = candidates_mask & -candidates_mask
            p = p_bit.bit_length() - 1
            
            # Option 1: Exclude p
            search(current_mask, forbidden_mask, candidates_mask ^ p_bit, current_size)
            
            # Option 2: Include p
            # Check if adding p creates a triple with existing points in current_mask
            # A triple is formed if there are two points a, b in current_mask such that {a, b, p} is a triple.
            # Since it's linear, for each pair (a, p) in current_mask, check if triple exists.
            # Actually, we can check: for each triple containing p, if the other 2 points are in current_mask, it's invalid.
            
            valid = True
            new_forbidden = forbidden_mask
            for t_idx in point_to_triples[p]:
                t = triples[t_idx]
                other1 = t[0] if t[0] != p else (t[1] if t[1] != p else t[2])
                other2 = t[1] if t[1] != p else (t[2] if t[2] != p else t[0])
                # Ensure we have distinct others
                others = [x for x in t if x != p]
                a, b = others[0], others[1]
                
                # If both a and b are in current_mask, adding p completes the triple.
                if (current_mask & (1 << a)) and (current_mask & (1 << b)):
                    valid = False
                    break
                # If one is in current_mask, the other is forbidden
                if (current_mask & (1 << a)):
                    new_forbidden |= (1 << b)
                if (current_mask & (1 << b)):
                    new_forbidden |= (1 << a)
            
            if valid:
                # Update candidates: remove p
                new_candidates = candidates_mask ^ p_bit
                # Remove forbidden points from candidates
                new_candidates &= ~new_forbidden
                # Note: We must also ensure that future additions don't conflict with each other?
                # The recursion handles that.
                search(current_mask | p_bit, new_forbidden, new_candidates, current_size + 1)

        search(0, 0, (1 << N) - 1, 0)
        return best_alpha

    def generate_initial_system():
        """Generate a random linear triple system."""
        triples = []
        used_pairs = set()
        
        # Create all possible triples
        all_triples = list(combinations(range(N), 3))
        random.shuffle(all_triples)
        
        for t in all_triples:
            # Check if any pair in t is already used
            conflict = False
            for p in combinations(t, 2):
                if p in used_pairs:
                    conflict = True
                    break
            if not conflict:
                triples.append(list(t))
                for p in combinations(t, 2):
                    used_pairs.add(p)
                    
        return triples

    def local_search_sa(triples, max_time):
        """Simulated Annealing to minimize independence number."""
        current_triples = triples[:]
        current_alpha = calculate_alpha(current_triples)
        best_triples = current_triples[:]
        best_alpha = current_alpha
        
        # Track used pairs for fast linearity check
        used_pairs = set()
        for t in current_triples:
            for p in combinations(t, 2):
                used_pairs.add(p)
        
        timeout = time.time() + max_time
        T = 1.0
        cooling = 0.999
        
        while time.time() < timeout:
            if T < 0.01:
                break
                
            move_type = random.choice(['add', 'remove', 'swap'])
            
            if move_type == 'add':
                # Find a random triple that doesn't conflict
                # Sample a few random triples
                added = False
                for _ in range(10):
                    t = list(random.sample(range(N), 3))
                    t.sort()
                    # Check pairs
                    ok = True
                    new_pairs = []
                    for p in combinations(t, 2):
                        if p in used_pairs:
                            ok = False
                            break
                        new_pairs.append(p)
                    if ok:
                        # Apply move
                        current_triples.append(t)
                        for p in new_pairs:
                            used_pairs.add(p)
                        
                        new_alpha = calculate_alpha(current_triples)
                        delta = new_alpha - current_alpha
                        if delta < 0 or random.random() < pow(2.71828, -delta / T):
                            current_alpha = new_alpha
                            if new_alpha < best_alpha:
                                best_alpha = new_alpha
                                best_triples = current_triples[:]
                        else:
                            # Revert
                            current_triples.pop()
                            for p in new_pairs:
                                used_pairs.discard(p)
                        added = True
                        break
                if not added:
                    continue
                    
            elif move_type == 'remove':
                if not current_triples:
                    continue
                idx = random.randint(0, len(current_triples) - 1)
                t = current_triples[idx]
                current_triples.pop(idx)
                for p in combinations(t, 2):
                    used_pairs.discard(p)
                
                new_alpha = calculate_alpha(current_triples)
                delta = new_alpha - current_alpha
                if delta < 0 or random.random() < pow(2.71828, -delta / T):
                    current_alpha = new_alpha
                    if new_alpha < best_alpha:
                        best_alpha = new_alpha
                        best_triples = current_triples[:]
                else:
                    # Revert
                    current_triples.insert(idx, t)
                    for p in combinations(t, 2):
                        used_pairs.add(p)
                        
            elif move_type == 'swap':
                if not current_triples:
                    continue
                idx = random.randint(0, len(current_triples) - 1)
                old_t = current_triples[idx]
                
                # Generate new triple
                for _ in range(10):
                    new_t = list(random.sample(range(N), 3))
                    new_t.sort()
                    # Check if new_t conflicts with others
                    ok = True
                    # Temporarily remove old_t pairs
                    temp_used = set(used_pairs)
                    for p in combinations(old_t, 2):
                        temp_used.discard(p)
                    
                    new_pairs = []
                    for p in combinations(new_t, 2):
                        if p in temp_used:
                            ok = False
                            break
                        new_pairs.append(p)
                    
                    if ok:
                        # Apply
                        current_triples[idx] = new_t
                        for p in combinations(old_t, 2):
                            used_pairs.discard(p)
                        for p in new_pairs:
                            used_pairs.add(p)
                        
                        new_alpha = calculate_alpha(current_triples)
                        delta = new_alpha - current_alpha
                        if delta < 0 or random.random() < pow(2.71828, -delta / T):
                            current_alpha = new_alpha
                            if new_alpha < best_alpha:
                                best_alpha = new_alpha
                                best_triples = current_triples[:]
                        else:
                            # Revert
                            current_triples[idx] = old_t
                            for p in combinations(old_t, 2):
                                used_pairs.add(p)
                            for p in new_pairs:
                                used_pairs.discard(p)
                        break
                T *= cooling

        return best_triples, best_alpha

    # Generate initial system
    initial = generate_initial_system()
    
    # Ensure linearity (should be by construction)
    if not is_linear(initial):
         # Fallback
         initial = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11], [12, 13, 14]]
         
    best_triples = initial
    best_score = calculate_alpha(best_triples)
    
    # Run local search with remaining time
    remaining_time = deadline - time.time() - 5
    if remaining_time > 5:
        best_triples, best_score = local_search_sa(best_triples, max_time=remaining_time)
    
    # Final check
    if not is_linear(best_triples):
        return [[3*i, 3*i+1, 3*i+2] for i in range(5)]
    
    return best_triples
