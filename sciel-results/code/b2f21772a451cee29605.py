import time
import random
import itertools
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 118  # Safety margin before 120s limit (evaluator kills at 150s)
    
    N = 16
    all_triples = list(itertools.combinations(range(N), 3))
    num_triples = len(all_triples)
    
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}
    
    # Precompute all K(2,2,2) configurations
    configs = []
    config_id_map = {}
    triples_in_config = []
    
    points = range(N)
    for pts in itertools.combinations(points, 6):
        p0, p1, p2, p3, p4, p5 = pts
        pairs = [
            [(p0, p1), (p2, p3), (p4, p5)],
            [(p0, p1), (p2, p4), (p3, p5)],
            [(p0, p1), (p2, p5), (p3, p4)],
            [(p0, p2), (p1, p3), (p4, p5)],
            [(p0, p2), (p1, p4), (p3, p5)],
            [(p0, p2), (p1, p5), (p3, p4)],
            [(p0, p3), (p1, p2), (p4, p5)],
            [(p0, p3), (p1, p4), (p2, p5)],
            [(p0, p3), (p1, p5), (p2, p4)],
            [(p0, p4), (p1, p2), (p3, p5)],
            [(p0, p4), (p1, p3), (p2, p5)],
            [(p0, p4), (p1, p5), (p2, p3)],
            [(p0, p5), (p1, p2), (p3, p4)],
            [(p0, p5), (p1, p3), (p2, p4)],
            [(p0, p5), (p1, p4), (p2, p3)]
        ]
        for pair_set in pairs:
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
    
    # Precompute: for each triple, which configs it belongs to
    triple_to_configs = [[] for _ in range(num_triples)]
    for cid, tris in enumerate(triples_in_config):
        for t_idx in tris:
            triple_to_configs[t_idx].append(cid)
            
    # Precompute: for each config, the 8 triples as a numpy array
    config_triples_arr = [np.array(tris, dtype=np.int32) for tris in triples_in_config]
    
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

    def greedy_construct(order=None):
        """Fast greedy construction with optional custom order."""
        present[:] = False
        config_counts[:] = 0
        current_set = set()
        
        if order is None:
            order = list(range(num_triples))
            random.shuffle(order)
        
        for t_idx in order:
            if is_valid_addition(t_idx):
                add_triple(t_idx)
                current_set.add(t_idx)
        
        return len(current_set), set(current_set)

    def local_search(current_set, max_iters=10000):
        """Targeted local search: try swaps to improve count."""
        present[:] = False
        config_counts[:] = 0
        for t_idx in current_set:
            present[t_idx] = True
            for cid in triple_to_configs[t_idx]:
                config_counts[cid] += 1
        
        count = len(current_set)
        best_count = count
        best_set_snapshot = set(current_set)
        
        # Build list of missing triples
        missing = [i for i in range(num_triples) if not present[i]]
        
        for iteration in range(max_iters):
            if time.time() > deadline - 5:
                break
                
            if count > best_count:
                best_count = count
                best_set_snapshot = set(current_set)
                
            if not missing:
                break
                
            # Try to add a random missing triple
            t_idx = random.choice(missing)
            
            if is_valid_addition(t_idx):
                add_triple(t_idx)
                current_set.add(t_idx)
                missing.remove(t_idx)
                count += 1
            else:
                # Blocked - try to find a swap
                conflicts = [cid for cid in triple_to_configs[t_idx] if config_counts[cid] == 7]
                if not conflicts:
                    continue
                    
                # Try swapping: remove a triple from a conflicting config, then add t_idx
                improved = False
                for cid in conflicts:
                    present_in_cid = [t for t in triples_in_config[cid] if present[t]]
                    if not present_in_cid:
                        continue
                        
                    # Try each candidate removal
                    for u_idx in present_in_cid:
                        # Check if removing u_idx and adding t_idx is valid
                        # First, check if u_idx is the only blocker for t_idx in this config
                        # Actually, just try the swap
                        remove_triple(u_idx)
                        current_set.remove(u_idx)
                        
                        if is_valid_addition(t_idx):
                            add_triple(t_idx)
                            current_set.add(t_idx)
                            # u_idx is now missing, t_idx is present
                            if u_idx not in missing:
                                missing.append(u_idx)
                            if t_idx in missing:
                                missing.remove(t_idx)
                            improved = True
                            break
                        else:
                            # Restore u_idx
                            add_triple(u_idx)
                            current_set.add(u_idx)
                    
                    if improved:
                        break
                
                # Random kick: remove 1-2 random triples and try to refill
                if not improved and random.random() < 0.1:
                    kick_size = random.randint(1, 2)
                    to_remove = random.sample(list(current_set), min(kick_size, len(current_set)))
                    for t in to_remove:
                        remove_triple(t)
                        current_set.remove(t)
                        count -= 1
                        if t not in missing:
                            missing.append(t)
                    
                    # Try to add back from missing
                    random.shuffle(missing)
                    for m in missing[:5]:
                        if is_valid_addition(m):
                            add_triple(m)
                            current_set.add(m)
                            count += 1
                            missing.remove(m)
        
        return best_count, best_set_snapshot

    def get_affine_lines():
        """Get affine lines in F2^4 as a good initial set."""
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
                    lines.append(triple_to_idx[t])
        return lines

    best_count = 0
    best_set = set()
    
    # Phase 1: Multiple greedy constructions with different strategies
    num_greedy_tries = 30
    for i in range(num_greedy_tries):
        if time.time() > deadline - 20:
            break
        
        # Vary the order: sometimes start with affine lines, sometimes random
        if i % 3 == 0:
            # Start with affine lines, then add random
            affine = get_affine_lines()
            remaining = [i for i in range(num_triples) if i not in set(affine)]
            random.shuffle(remaining)
            order = affine + remaining
        elif i % 3 == 1:
            order = list(range(num_triples))
            random.shuffle(order)
        else:
            # Weighted: favor triples that appear in fewer configs
            weights = [1.0 / (len(triple_to_configs[i]) + 1) for i in range(num_triples)]
            order = list(range(num_triples))
            random.shuffle(order)
            # Sort by weight with some randomness
            order = sorted(order, key=lambda x: random.random() * weights[x])
        
        count, s = greedy_construct(order)
        if count > best_count:
            best_count = count
            best_set = s
    
    # Phase 2: Local search on the best solution
    if time.time() < deadline - 10:
        count, s = local_search(best_set, max_iters=50000)
        if count > best_count:
            best_count = count
            best_set = s
    
    # Phase 3: More greedy + local search cycles
    num_cycles = 10
    for i in range(num_cycles):
        if time.time() > deadline - 15:
            break
        
        # Random greedy start
        order = list(range(num_triples))
        random.shuffle(order)
        count, s = greedy_construct(order)
        
        # Quick local search
        if time.time() < deadline - 10:
            count2, s2 = local_search(s, max_iters=20000)
            if count2 > best_count:
                best_count = count2
                best_set = s2
    
    result = [all_triples[i] for i in best_set]
    
    return result
