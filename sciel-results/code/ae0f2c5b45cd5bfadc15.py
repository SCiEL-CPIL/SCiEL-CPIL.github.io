import random
import itertools
import numpy as np

def construct():
    points = list(range(16))
    all_triples_list = list(itertools.combinations(points, 3))
    n_triples = len(all_triples_list)
    
    # Map triple to index
    triple_to_idx = {t: i for i, t in enumerate(all_triples_list)}
    
    # Generate all K(2,2,2) configurations
    # A configuration is defined by 3 disjoint pairs {a,a'}, {b,b'}, {c,c'}
    # The 8 transversals are (a or a', b or b', c or c')
    
    all_configs_set = set()
    
    # Precompute matchings for 6 points (indices 0..5)
    matchings = []
    def gen_matchings(remaining, current):
        if not remaining:
            matchings.append(current[:])
            return
        p = remaining[0]
        for i in range(1, len(remaining)):
            q = remaining[i]
            new_remaining = remaining[1:i] + remaining[i+1:]
            current.append((p, q))
            gen_matchings(new_remaining, current)
            current.pop()
            
    gen_matchings(list(range(6)), [])
    
    for six_pts in itertools.combinations(range(16), 6):
        for m in matchings:
            pairs = []
            for u, v in m:
                pairs.append((six_pts[u], six_pts[v]))
            
            p0, p1, p2 = pairs
            triples_in_config = set()
            for i in (0, 1):
                for j in (0, 1):
                    for k in (0, 1):
                        x = p0[i]
                        y = p1[j]
                        z = p2[k]
                        t = tuple(sorted((x, y, z)))
                        triples_in_config.add(triple_to_idx[t])
            
            cfg_key = frozenset(triples_in_config)
            all_configs_set.add(cfg_key)
            
    configs = list(all_configs_set)
    n_configs = len(configs)
    
    # Build reverse index: for each triple, which configs contain it?
    configs_containing = [[] for _ in range(n_triples)]
    for i, cfg in enumerate(configs):
        for t_idx in cfg:
            configs_containing[t_idx].append(i)
            
    # Simulated Annealing
    best_S = [0] * n_triples
    best_score = -1
    
    for restart in range(20):
        random.seed(restart)
        S = [random.randint(0, 1) for _ in range(n_triples)]
        
        counts = np.zeros(n_configs, dtype=np.int32)
        for j in range(n_triples):
            if S[j]:
                for c_idx in configs_containing[j]:
                    counts[c_idx] += 1
                    
        T = 100.0
        cooling = 0.95
        P = 10
        
        for step in range(5000):
            j = random.randint(0, n_triples - 1)
            
            if S[j]:
                viol_removed = 0
                for c_idx in configs_containing[j]:
                    if counts[c_idx] == 8:
                        viol_removed += 1
                        
                delta_E = -1 + P * viol_removed
                
                if delta_E < 0 or random.random() < np.exp(delta_E / T):
                    S[j] = 0
                    for c_idx in configs_containing[j]:
                        counts[c_idx] -= 1
            else:
                viol_added = 0
                for c_idx in configs_containing[j]:
                    if counts[c_idx] == 7:
                        viol_added += 1
                        
                delta_E = 1 - P * viol_added
                
                if delta_E > 0 or random.random() < np.exp(delta_E / T):
                    S[j] = 1
                    for c_idx in configs_containing[j]:
                        counts[c_idx] += 1
            
            T *= cooling
        
        # Cleanup: remove triples to ensure validity
        changed = True
        while changed:
            changed = False
            for c_idx in range(n_configs):
                if counts[c_idx] >= 8:
                    candidates = [t for t in configs[c_idx] if S[t]]
                    if candidates:
                        t_remove = random.choice(candidates)
                        S[t_remove] = 0
                        for cc in configs_containing[t_remove]:
                            counts[cc] -= 1
                        changed = True
                        break
        
        current_size = sum(S)
        if current_size > best_score:
            best_score = current_size
            best_S = S[:]
            
    result = []
    for j in range(n_triples):
        if best_S[j]:
            result.append(list(all_triples_list[j]))
            
    return result
