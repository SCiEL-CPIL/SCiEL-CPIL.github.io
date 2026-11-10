import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    points = list(range(16))
    all_triples = list(combinations(points, 3))
    n_triples = len(all_triples)
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}
    
    # Precompute pairs for each triple
    triple_pairs = []
    for t in all_triples:
        triple_pairs.append([(t[0], t[1]), (t[0], t[2]), (t[1], t[2])])
        
    # Map pair to list of triple indices
    pair_to_triples = {}
    for i, t in enumerate(all_triples):
        for p in triple_pairs[i]:
            if p not in pair_to_triples:
                pair_to_triples[p] = []
            pair_to_triples[p].append(i)

    def can_add(triple_idx, used_pairs):
        for p in triple_pairs[triple_idx]:
            if p in used_pairs:
                return False
        return True

    def greedy_init():
        used_pairs = set()
        current_triples = []
        indices = list(range(n_triples))
        random.shuffle(indices)
        for i in indices:
            if can_add(i, used_pairs):
                current_triples.append(i)
                for p in triple_pairs[i]:
                    used_pairs.add(p)
        return current_triples

    def calc_alpha(triples):
        # Check all subsets? Too slow for 16 vertices (2^16 = 65536).
        # But we can check descending sizes.
        # Max size 16.
        # For K from 16 down to 1:
        #   Check if any subset of size K is independent.
        #   If yes, alpha = K.
        #   To check if ANY subset of size K is independent:
        #   We can use the same counting method.
        #   If count of independent sets of size K > 0, then alpha >= K.
        #   So find max K such that count > 0.
        
        # This is expensive if done naively.
        # But for 16 vertices, we can just do it.
        # However, we only need this for initialization to pick starting K.
        # We can approximate or just start at K=7 and go down.
        
        # Let's just start SA at K=7. If we can't get 0, we keep the solution.
        # We will track the best solution found by actual alpha calculation at the end?
        # No, the proxy is good enough.
        # But to be safe, we should verify the final solution's alpha.
        
        pass # Placeholder

    # Precompute K-sets for a given K
    def get_ksets_data(K):
        all_ksets = list(combinations(points, K))
        n_ksets = len(all_ksets)
        kset_to_idx = {s: i for i, s in enumerate(all_ksets)}
        
        # triple_idx -> list of kset_indices
        triple_to_ksets = [[] for _ in range(n_triples)]
        for k_idx, kset in enumerate(all_ksets):
            # Find triples in this kset
            # kset has K elements. Triples are subsets of size 3.
            for t in combinations(kset, 3):
                t_idx = triple_to_idx[t]
                triple_to_ksets[t_idx].append(k_idx)
                
        return all_ksets, triple_to_ksets

    def optimize_K(initial_triples, K, time_limit):
        if K < 3:
            return initial_triples, 0 # Alpha < 3 is impossible for 16 pts with triples? 
                                     # If alpha < 3, every pair is in a triple? No.
                                     # If alpha = 2, no independent set of size 3.
                                     # This means every triple of points contains an edge?
                                     # No, independent set of size 3 means 3 points with no triple among them.
                                     # If alpha <= 2, every 3 points contain a triple.
                                     # This is a Steiner Triple System property (packing).
                                     # With 16 points, max edges 40. Pairs 120.
                                     # 40*3 = 120 pairs. So STS(16) would cover all pairs.
                                     # But STS(16) doesn't exist.
                                     # So alpha >= 3.
                                     # Actually, if we can't cover all pairs, there is a pair not in any triple.
                                     # That pair is an independent set of size 2.
                                     # So alpha is at least 2.
                                     # Can alpha be 2?
                                     # If alpha=2, no 3 points are independent.
                                     # This implies every 3 points contain a triple.
                                     # This is a 3-(16,3,1) design? No, just that every triple of vertices contains an edge of the hypergraph.
                                     # This requires the hypergraph to be a "blocking set" for all triples.
                                     # Max edges 40. Total triples 560.
                                     # Each edge blocks C(13,0)=1 triple? No.
                                     # An edge {a,b,c} is contained in itself.
                                     # Does it block other triples? No, independence is about *containing* an edge.
                                     # A set S is independent if it contains NO edge.
                                     # So if every 3-set contains an edge, then alpha <= 2.
                                     # This is equivalent to the hypergraph covering all 3-sets.
                                     # This is impossible with 40 edges (each covers 1 triple? No).
                                     # Wait. An edge is a triple.
                                     # "S contains an edge" means there exists an edge E such that E subset S.
                                     # If |S|=3, S contains an edge iff S is an edge.
                                     # So if alpha <= 2, it means there is NO set of 3 points that is NOT an edge.
                                     # i.e., Every 3-set is an edge.
                                     # This requires 560 edges. Max is 40.
                                     # So alpha >= 3.
        
        all_ksets, triple_to_ksets = get_ksets_data(K)
        n_ksets = len(all_ksets)
        
        # Initialize state
        current_triples = list(initial_triples)
        used_pairs = set()
        for t_idx in current_triples:
            for p in triple_pairs[t_idx]:
                used_pairs.add(p)
                
        counts = [0] * n_ksets
        for t_idx in current_triples:
            for k_idx in triple_to_ksets[t_idx]:
                counts[k_idx] += 1
                
        indep_count = sum(1 for c in counts if c == 0)
        
        best_triples = list(current_triples)
        best_indep = indep_count
        
        T = 1.0
        cooling = 0.9995
        min_T = 0.01
        
        # Precompute list of valid moves? No, dynamic.
        
        while time.time() < time_limit and T > min_T:
            # Move: Remove t_old, Add t_new
            if not current_triples:
                # Just add random
                t_new = random.randint(0, n_triples - 1)
                if can_add(t_new, used_pairs):
                    # Apply
                    current_triples.append(t_new)
                    for p in triple_pairs[t_new]:
                        used_pairs.add(p)
                    
                    # Update counts
                    delta = 0
                    for k_idx in triple_to_ksets[t_new]:
                        if counts[k_idx] == 0:
                            delta -= 1
                        counts[k_idx] += 1
                    indep_count += delta
                else:
                    T *= cooling
                    continue
            else:
                t_old_idx = random.randint(0, len(current_triples) - 1)
                t_old = current_triples[t_old_idx]
                
                # Try to find replacement
                found = False
                for _ in range(5): # Few attempts per step
                    t_new = random.randint(0, n_triples - 1)
                    if t_new == t_old:
                        continue
                    
                    # Check validity of swap
                    # Remove t_old pairs
                    pairs_old = triple_pairs[t_old]
                    for p in pairs_old:
                        used_pairs.discard(p)
                    
                    if can_add(t_new, used_pairs):
                        # Valid swap
                        # Calculate delta
                        delta = 0
                        # Remove t_old
                        for k_idx in triple_to_ksets[t_old]:
                            counts[k_idx] -= 1
                            if counts[k_idx] == 0:
                                delta += 1
                        # Add t_new
                        for k_idx in triple_to_ksets[t_new]:
                            if counts[k_idx] == 0:
                                delta -= 1
                            counts[k_idx] += 1
                        
                        # Acceptance
                        if delta < 0 or random.random() < pow(2.71828, -delta / T):
                            # Accept
                            current_triples[t_old_idx] = t_new
                            for p in triple_pairs[t_new]:
                                used_pairs.add(p)
                            indep_count += delta
                            
                            if indep_count < best_indep:
                                best_indep = indep_count
                                best_triples = list(current_triples)
                                if best_indep == 0:
                                    return best_triples, 0
                            
                            found = True
                            break
                    else:
                        # Restore t_old pairs
                        for p in pairs_old:
                            used_pairs.add(p)
                
                if not found:
                    # State is restored (pairs_old re-added)
                    T *= cooling
                    continue
            
            T *= cooling
            
        return best_triples, best_indep

    # Main
    best_solution = greedy_init()
    
    # Calculate initial alpha roughly?
    # Just start K=7.
    # If we achieve 0 for K=7, alpha <= 6.
    # Then try K=6.
    
    current_triples = best_solution
    
    for K in range(7, 2, -1):
        if time.time() > deadline - 5:
            break
            
        # Run optimization for this K
        # Give it a time budget
        # If we have 100s left, give 50s to K=7, 30s to K=6, etc?
        # Or just run until time limit for each, but break if 0 found.
        
        # Better: Run SA for a fixed number of iterations or time slice.
        # Let's allocate time based on remaining time.
        # If K=7, try for 30s. If 0 found, move to 6.
        # If not 0, keep best, move to 6?
        # Actually, if we can't get 0 for K=7, we might still improve K=6?
        # No, if alpha >= 7, we can't have alpha <= 5.
        # So we should focus on the current boundary.
        
        # Let's just run until time limit for the highest K we haven't solved.
        # But we might get stuck.
        
        # Strategy:
        # 1. Try K=7 for 40s.
        # 2. If success (0 indep), set K=6.
        # 3. Try K=6 for 30s.
        # 4. If success, set K=5.
        # 5. Try K=5 for 20s.
        # 6. Return best.
        
        pass # Implement specific time budgets

    return [list(all_triples[i]) for i in current_triples]
