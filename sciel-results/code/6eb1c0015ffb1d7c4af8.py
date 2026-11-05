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

    def walk_sat_search(initial_set, max_iters=50000):
        """
        WalkSAT-style stochastic local search for Max-SAT formulation.
        Variables: whether each triple is included (1) or not (0).
        Objective: maximize number of included triples.
        Constraint: no K(2,2,2) configuration has all 8 triples included.
        
        This is a Max-SAT where we have:
        - Positive unit clauses for each triple (want to include it)
        - Negative clauses: for each config, NOT(all 8 present)
        
        We use a simplified approach: 
        - Start from initial set
        - Repeatedly try to add random missing triples
        - If blocked, remove a random conflicting triple and retry
        - Track best solution found
        """
        present[:] = False
        config_counts[:] = 0
        current_set = set()
        
        # Initialize
        for t in initial_set:
            if is_valid_addition(triple_to_idx[t]):
                add_triple(triple_to_idx[t])
                current_set.add(triple_to_idx[t])
                
        count = len(current_set)
        best_count = count
        best_set_snapshot = set(current_set)
        
        random.seed(int(time.time() * 10000) % 100000)
        
        # WalkSAT parameters
        noise = 0.1  # Probability of random flip vs greedy
        flip_prob = 0.3  # Probability to flip an existing variable
        
        for iteration in range(max_iters):
            if time.time() > deadline - 5:
                break
                
            if count > best_count:
                best_count = count
                best_set_snapshot = set(current_set)
                
            # Try to add a random missing triple
            missing = [i for i in range(num_triples) if not present[i]]
            if not missing:
                break
                
            t_idx = random.choice(missing)
            
            if is_valid_addition(t_idx):
                add_triple(t_idx)
                current_set.add(t_idx)
                count += 1
            else:
                # Blocked - need to resolve conflict
                # Find a conflicting config
                conflicts = []
                for cid in triple_to_configs[t_idx]:
                    if config_counts[cid] == 7:
                        conflicts.append(cid)
                        
                if not conflicts:
                    continue
                    
                # With some probability, do a random kick
                if random.random() < noise:
                    # Random kick: remove a random existing triple
                    if current_set:
                        u_idx = random.choice(list(current_set))
                        remove_triple(u_idx)
                        current_set.remove(u_idx)
                        count -= 1
                        # Try to add the original triple
                        if is_valid_addition(t_idx):
                            add_triple(t_idx)
                            current_set.add(t_idx)
                            count += 1
                else:
                    # Greedy: remove a triple from the most constrained config
                    cid = random.choice(conflicts)
                    present_in_cid = [t for t in triples_in_config[cid] if present[t]]
                    if present_in_cid:
                        # Remove the triple that appears in the most configs
                        u_idx = max(present_in_cid, key=lambda x: len(triple_to_configs[x]))
                        remove_triple(u_idx)
                        current_set.remove(u_idx)
                        count -= 1
                        if is_valid_addition(t_idx):
                            add_triple(t_idx)
                            current_set.add(t_idx)
                            count += 1
        
        return best_count, list(best_set_snapshot)

    def simulated_annealing(initial_set, max_iters=50000):
        """
        Simulated annealing approach.
        """
        present[:] = False
        config_counts[:] = 0
        current_set = set()
        
        for t in initial_set:
            if is_valid_addition(triple_to_idx[t]):
                add_triple(triple_to_idx[t])
                current_set.add(triple_to_idx[t])
                
        count = len(current_set)
        best_count = count
        best_set_snapshot = set(current_set)
        
        random.seed(int(time.time() * 10000) % 100000)
        
        T = 1.0
        cooling_rate = 0.9999
        min_T = 0.01
        
        for iteration in range(max_iters):
            if time.time() > deadline - 5:
                break
                
            if count > best_count:
                best_count = count
                best_set_snapshot = set(current_set)
                
            # Choose a move: add, remove, or swap
            move_type = random.random()
            
            if move_type < 0.5:
                # Try to add a random missing triple
                missing = [i for i in range(num_triples) if not present[i]]
                if not missing:
                    continue
                t_idx = random.choice(missing)
                
                if is_valid_addition(t_idx):
                    add_triple(t_idx)
                    current_set.add(t_idx)
                    count += 1
                else:
                    # Blocked - decide whether to accept based on temperature
                    conflicts = [cid for cid in triple_to_configs[t_idx] if config_counts[cid] == 7]
                    if conflicts:
                        cid = random.choice(conflicts)
                        present_in_cid = [t for t in triples_in_config[cid] if present[t]]
                        if present_in_cid:
                            u_idx = random.choice(present_in_cid)
                            # Remove u, add t
                            remove_triple(u_idx)
                            current_set.remove(u_idx)
                            if is_valid_addition(t_idx):
                                add_triple(t_idx)
                                current_set.add(t_idx)
                                # count unchanged (swap)
                            else:
                                add_triple(u_idx)
                                current_set.add(u_idx)
            elif move_type < 0.7:
                # Try to remove a random existing triple and add another
                if current_set:
                    u_idx = random.choice(list(current_set))
                    remove_triple(u_idx)
                    current_set.remove(u_idx)
                    count -= 1
                    
                    missing = [i for i in range(num_triples) if not present[i]]
                    if missing:
                        t_idx = random.choice(missing)
                        if is_valid_addition(t_idx):
                            add_triple(t_idx)
                            current_set.add(t_idx)
                            count += 1
                        else:
                            # Restore
                            add_triple(u_idx)
                            current_set.add(u_idx)
                            count += 1
            else:
                # Random kick: remove 1-3 random triples
                kick_size = random.randint(1, 3)
                to_remove = random.sample(list(current_set), min(kick_size, len(current_set)))
                for t in to_remove:
                    remove_triple(t)
                    current_set.remove(t)
                    count -= 1
                
                # Try to add back
                missing = [i for i in range(num_triples) if not present[i]]
                random.shuffle(missing)
                for t_idx in missing[:10]:
                    if is_valid_addition(t_idx):
                        add_triple(t_idx)
                        current_set.add(t_idx)
                        count += 1
            
            T *= cooling_rate
            if T < min_T:
                T = min_T
                
        return best_count, list(best_set_snapshot)

    # Get initial lines from affine geometry
    initial_lines = get_initial_lines()
    
    best_count = 0
    best_set = []
    
    # Run multiple methods and take the best
    methods = [
        ("walksat", walk_sat_search),
        ("anneal", simulated_annealing),
    ]
    
    for method_name, method_func in methods:
        if time.time() > deadline - 10:
            break
            
        count, s = method_func(initial_lines)
        if count > best_count:
            best_count = count
            best_set = s
            
    # If time remains, try with different random seeds
    num_extra_tries = 0
    while time.time() < deadline - 10 and num_extra_tries < 5:
        # Random greedy start
        present[:] = False
        config_counts[:] = 0
        current_set = set()
        order = list(range(num_triples))
        random.shuffle(order)
        for t_idx in order:
            if is_valid_addition(t_idx):
                add_triple(t_idx)
                current_set.add(t_idx)
        
        init_triples = [all_triples[i] for i in current_set]
        
        # Try walksat from this start
        count2, s2 = walk_sat_search(init_triples, max_iters=20000)
        if count2 > best_count:
            best_count = count2
            best_set = s2
            
        num_extra_tries += 1
        
    result = [all_triples[i] for i in best_set]
    
    return result
