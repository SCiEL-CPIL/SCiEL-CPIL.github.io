import time
import random
import itertools
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    N = 16
    all_triples = list(itertools.combinations(range(N), 3))
    num_triples = len(all_triples)
    
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}
    
    # Precompute all K(2,2,2) configurations
    configs = []
    config_id_map = {}
    triples_in_config = []
    
    points = range(N)
    # Generate all 6-point subsets
    # For each 6-point subset, there are 15 ways to partition into 3 pairs
    # Each partition defines a K(2,2,2)
    # We iterate over all 6-point subsets to be exhaustive
    
    for pts in itertools.combinations(points, 6):
        p0, p1, p2, p3, p4, p5 = pts
        # 15 pairings of 6 elements into 3 pairs
        # We can generate these by fixing p0 with p1, p2, p3, p4, p5
        # If p0-p1, remaining 4 have 3 pairings
        # If p0-p2, remaining 4 have 3 pairings
        # If p0-p3, remaining 4 have 3 pairings
        # If p0-p4, remaining 4 have 3 pairings
        # If p0-p5, remaining 4 have 3 pairings
        
        pair_sets = []
        
        # Case 1: (p0, p1)
        rem = [p2, p3, p4, p5]
        pair_sets.append([(p0, p1), (rem[0], rem[1]), (rem[2], rem[3])])
        pair_sets.append([(p0, p1), (rem[0], rem[2]), (rem[1], rem[3])])
        pair_sets.append([(p0, p1), (rem[0], rem[3]), (rem[1], rem[2])])
        
        # Case 2: (p0, p2)
        rem = [p1, p3, p4, p5]
        pair_sets.append([(p0, p2), (rem[0], rem[1]), (rem[2], rem[3])])
        pair_sets.append([(p0, p2), (rem[0], rem[2]), (rem[1], rem[3])])
        pair_sets.append([(p0, p2), (rem[0], rem[3]), (rem[1], rem[2])])
        
        # Case 3: (p0, p3)
        rem = [p1, p2, p4, p5]
        pair_sets.append([(p0, p3), (rem[0], rem[1]), (rem[2], rem[3])])
        pair_sets.append([(p0, p3), (rem[0], rem[2]), (rem[1], rem[3])])
        pair_sets.append([(p0, p3), (rem[0], rem[3]), (rem[1], rem[2])])
        
        # Case 4: (p0, p4)
        rem = [p1, p2, p3, p5]
        pair_sets.append([(p0, p4), (rem[0], rem[1]), (rem[2], rem[3])])
        pair_sets.append([(p0, p4), (rem[0], rem[2]), (rem[1], rem[3])])
        pair_sets.append([(p0, p4), (rem[0], rem[3]), (rem[1], rem[2])])
        
        # Case 5: (p0, p5)
        rem = [p1, p2, p3, p4]
        pair_sets.append([(p0, p5), (rem[0], rem[1]), (rem[2], rem[3])])
        pair_sets.append([(p0, p5), (rem[0], rem[2]), (rem[1], rem[3])])
        pair_sets.append([(p0, p5), (rem[0], rem[3]), (rem[1], rem[2])])

        for pair_set in pair_sets:
            norm_pairs = tuple(sorted(tuple(sorted(p)) for p in pair_set))
            if norm_pairs not in config_id_map:
                cid = len(configs)
                config_id_map[norm_pairs] = cid
                configs.append(norm_pairs)
                
                (a,b), (c,d), (e,f) = norm_pairs
                tris = []
                for x in (a,b):
                    for y in (c,d):
                        for z in (e,f):
                            t = tuple(sorted((x,y,z)))
                            tris.append(triple_to_idx[t])
                triples_in_config.append(tris)
                
    num_configs = len(configs)
    
    # Map triple -> list of config indices
    triple_to_configs = [[] for _ in range(num_triples)]
    for cid, tris in enumerate(triples_in_config):
        for t_idx in tris:
            triple_to_configs[t_idx].append(cid)
            
    present = np.zeros(num_triples, dtype=bool)
    config_counts = np.zeros(num_configs, dtype=np.int8)
    
    def add_triple(t_idx):
        present[t_idx] = True
        for cid in triple_to_configs[t_idx]:
            config_counts[cid] += 1
            if config_counts[cid] == 8:
                return False
        return True
        
    def remove_triple(t_idx):
        present[t_idx] = False
        for cid in triple_to_configs[t_idx]:
            config_counts[cid] -= 1
        return True

    def is_valid_addition(t_idx):
        if present[t_idx]:
            return False
        for cid in triple_to_configs[t_idx]:
            if config_counts[cid] == 7:
                return False
        return True

    def get_initial_lines():
        # Affine lines in F2^4
        lines = []
        seen = set()
        for x in range(N):
            for y in range(N):
                if x == y: continue
                z = x ^ y
                if z == x or z == y: continue
                t = tuple(sorted((x, y, z)))
                if t not in seen:
                    seen.add(t)
                    lines.append(t)
        return lines

    def greedy_construction(order):
        present[:] = False
        config_counts[:] = 0
        current_set = set()
        for t_idx in order:
            if is_valid_addition(t_idx):
                add_triple(t_idx)
                current_set.add(t_idx)
        return current_set

    def local_search_2in1out(current_set, max_iters=100000):
        """
        Structural improvement: 2-in, 1-out swaps.
        Find a triple T to remove such that at least 2 new triples can be added.
        This escapes local maxima where no single addition is possible.
        """
        best_count = len(current_set)
        best_set = set(current_set)
        
        # Precompute for each triple, the set of triples that conflict with it
        # Conflict means: they share a config where count would become 8 if both added?
        # No, simpler: 
        # If we remove T, we lower counts in configs containing T.
        # A triple U (not present) can be added if for all c in configs(U), count[c] < 8.
        # After removing T, count[c] is lower for c in configs(T).
        # So U becomes addable if:
        # 1. U was not addable before (count[c] == 7 for some c in configs(U))
        # 2. AND that c is in configs(T) (so count drops to 6)
        # 3. AND no other config in configs(U) is at 7 or 8 (but if U wasn't addable, only one was at 7? 
        #    Actually, if U has two configs at 7, removing T only helps if T is in BOTH. 
        #    If T is in one, the other is still 7, so U still not addable.)
        
        # Efficient check:
        # For each T in current_set:
        #   Let C_T = set of configs containing T.
        #   For each U not in current_set:
        #      Let C_U = set of configs containing U.
        #      U is "blocked" by configs where count == 7.
        #      Let B_U = {c in C_U | count[c] == 7}.
        #      If B_U is empty, U is already addable (should have been added in greedy, or added later).
        #      If B_U is not empty, U becomes addable after removing T if B_U is a subset of C_T 
        #      AND for all c in C_U \ C_T, count[c] < 7.
        #      Actually, if c in C_U \ C_T has count 7, U is still blocked.
        #      So condition: B_U subset C_T AND max(count[c] for c in C_U if c not in C_T) < 7.
        
        # To make this fast, we can iterate T, then iterate candidates U.
        # But num_triples is 560. 560 * 560 = 313k checks per iteration. 
        # If we do this 1000 times, it's 300M operations. Might be slow in Python.
        
        # Optimization: 
        # Maintain a list of "blocked" triples and the reason (which config is at 7).
        # When we remove T, we update counts.
        
        # Let's try a simpler heuristic first: Randomized 2-in-1-out
        # 1. Pick random T in current_set.
        # 2. Remove T.
        # 3. Find all U not in current_set that are now valid.
        # 4. If count(Valid_U) >= 2, pick the best 2 (or all) and add them.
        # 5. If count > best_count, update. Else, restore.
        
        # We need a fast way to find valid U after removing T.
        # Instead of checking all 560, we can check only those U that were blocked by configs containing T.
        
        # Let's implement the check for a specific T removal.
        
        for _ in range(max_iters):
            if time.time() > deadline - 5:
                break
                
            if not current_set:
                break
                
            T = random.choice(list(current_set))
            
            # Temporarily remove T
            remove_triple(T)
            current_set.remove(T)
            
            # Find all U not in current_set that are now valid
            # We only need to check U that share a config with T? 
            # No, U must be valid in ALL its configs.
            # If U doesn't share a config with T, its validity didn't change.
            # So we only need to check U that share at least one config with T AND were previously invalid.
            
            candidates = []
            for cid in triple_to_configs[T]:
                # U must be in this config? No, U just needs to be affected by the count drop.
                # The count drop affects configs in triple_to_configs[T].
                # If U has a config c in triple_to_configs[T], and c was at 7, now it's 6.
                # So U might become valid.
                # Let's iterate all U not in current_set? That's expensive.
                
                # Alternative: Iterate all U not in current_set, but filter quickly.
                pass
            
            # Simpler: Just iterate all missing triples. 560 is small.
            missing = [i for i in range(num_triples) if not present[i]]
            valid_adds = []
            for u_idx in missing:
                if is_valid_addition(u_idx):
                    valid_adds.append(u_idx)
            
            # If we can add at least 2, do it
            if len(valid_adds) >= 2:
                # Add the top 2 (or more if available and valid)
                # We should check if adding them together is valid.
                # Since they are all individually valid after T removal, 
                # adding one might block another.
                # Greedy add:
                added = 0
                for u_idx in valid_adds:
                    if is_valid_addition(u_idx):
                        add_triple(u_idx)
                        current_set.add(u_idx)
                        added += 1
                        if added >= 2:
                            break
                
                if len(current_set) > best_count:
                    best_count = len(current_set)
                    best_set = set(current_set)
                else:
                    # Revert: Remove added ones, restore T
                    # This is tricky. Let's just keep the state if it's not worse, 
                    # but we want to explore. 
                    # If we didn't improve, revert to T?
                    # Actually, if we added 2 and removed 1, count increased by 1.
                    # If len(current_set) == best_count, it's a tie.
                    # If we added 2, count is +1. So it must be > best_count if best_count was previous.
                    # Wait, if best_count was updated, we keep it.
                    # If we added 2 but it didn't beat best_count (e.g. best_count was already higher), 
                    # we should probably revert to T to not lose the T?
                    # No, if we added 2, we have T removed and 2 added. Net +1.
                    # If this new state is not better than global best, we can keep it as current state for further search?
                    # Or revert? 
                    # Let's keep it if it's valid. It's a different local minimum.
                    pass
            else:
                # Revert T
                add_triple(T)
                current_set.add(T)
                
        return best_count, best_set

    def local_search_1in1out(current_set, max_iters=100000):
        """
        Standard 1-in, 1-out swap.
        """
        best_count = len(current_set)
        best_set = set(current_set)
        
        for _ in range(max_iters):
            if time.time() > deadline - 5:
                break
                
            if not current_set:
                break
                
            T = random.choice(list(current_set))
            remove_triple(T)
            current_set.remove(T)
            
            missing = [i for i in range(num_triples) if not present[i]]
            valid_adds = []
            for u_idx in missing:
                if is_valid_addition(u_idx):
                    valid_adds.append(u_idx)
            
            if valid_adds:
                # Pick one
                u_idx = random.choice(valid_adds)
                add_triple(u_idx)
                current_set.add(u_idx)
                
                if len(current_set) > best_count:
                    best_count = len(current_set)
                    best_set = set(current_set)
                # Note: count is same, so best_count won't increase here unless we find a path to +1 later
                # But this moves us to a different local minimum.
            else:
                add_triple(T)
                current_set.add(T)
                
        return best_count, best_set

    # Get initial lines from affine geometry
    initial_lines = get_initial_lines()
    
    best_count = 0
    best_set = set()
    
    # 1. Greedy construction
    # Try a few different random orders for greedy
    for i in range(5):
        if time.time() > deadline - 30:
            break
        order = list(range(num_triples))
        random.shuffle(order)
        current_set = greedy_construction(order)
        
        # 2. Local search 2-in-1-out
        count, s = local_search_2in1out(current_set, max_iters=20000)
        if count > best_count:
            best_count = count
            best_set = s
            
        # 3. Local search 1-in-1-out
        if time.time() > deadline - 10:
            break
        count, s = local_search_1in1out(s, max_iters=20000)
        if count > best_count:
            best_count = count
            best_set = s
            
    # Final cleanup: Ensure best_set is valid and convert to list of triples
    # Re-verify best_set
    present[:] = False
    config_counts[:] = 0
    final_set = set()
    for t_idx in best_set:
        if is_valid_addition(t_idx):
            add_triple(t_idx)
            final_set.add(t_idx)
            
    result = [all_triples[i] for i in final_set]
    
    return result

if __name__ == "__main__":
    res = construct()
    print(f"Found {len(res)} triples")
