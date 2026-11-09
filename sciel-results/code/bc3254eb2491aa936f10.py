import time
import random
import itertools
import math

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit
    
    n = 16
    all_triples = list(itertools.combinations(range(n), 3))
    num_triples = len(all_triples)
    triple_index = {t: i for i, t in enumerate(all_triples)}
    
    # Pre-compute forbidden configs (hyperedges)
    forbidden_configs = []
    seen_configs = set()
    
    for six_points in itertools.combinations(range(n), 6):
        p0, p1, p2, p3, p4, p5 = six_points
        pairings = [
            [(p0,p1),(p2,p3),(p4,p5)],
            [(p0,p1),(p2,p4),(p3,p5)],
            [(p0,p1),(p2,p5),(p3,p4)],
            [(p0,p2),(p1,p3),(p4,p5)],
            [(p0,p2),(p1,p4),(p3,p5)],
            [(p0,p2),(p1,p5),(p3,p4)],
            [(p0,p3),(p1,p2),(p4,p5)],
            [(p0,p3),(p1,p4),(p2,p5)],
            [(p0,p3),(p1,p5),(p2,p4)],
            [(p0,p4),(p1,p2),(p3,p5)],
            [(p0,p4),(p1,p3),(p2,p5)],
            [(p0,p4),(p1,p5),(p2,p3)],
            [(p0,p5),(p1,p2),(p3,p4)],
            [(p0,p5),(p1,p3),(p2,p4)],
            [(p0,p5),(p1,p4),(p2,p3)],
        ]
        
        for pairs in pairings:
            (a1,b1), (a2,b2), (a3,b3) = pairs
            transversals = []
            for i in range(2):
                for j in range(2):
                    for k in range(2):
                        t = tuple(sorted([(a1,b1)[i], (a2,b2)[j], (a3,b3)[k]]))
                        transversals.append(triple_index[t])
            
            config = frozenset(transversals)
            if config not in seen_configs:
                seen_configs.add(config)
                forbidden_configs.append(config)
    
    num_configs = len(forbidden_configs)
    
    triple_to_configs = [[] for _ in range(num_triples)]
    config_to_triples = [list(c) for c in forbidden_configs]
    for ci, config in enumerate(forbidden_configs):
        for t in config:
            triple_to_configs[t].append(ci)

    def greedy_init():
        present = [False] * num_triples
        config_count = [0] * num_configs
        indices = list(range(num_triples))
        random.shuffle(indices)
        
        for t in indices:
            valid = True
            for ci in triple_to_configs[t]:
                if config_count[ci] == 7:
                    valid = False
                    break
            if valid:
                present[t] = True
                for ci in triple_to_configs[t]:
                    config_count[ci] += 1
        return present, config_count

    def local_search(present, config_count):
        missing = [t for t in range(num_triples) if not present[t]]
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            random.shuffle(missing)
            for t in missing:
                if present[t]:
                    continue
                safe = True
                for ci in triple_to_configs[t]:
                    if config_count[ci] == 7:
                        safe = False
                        break
                if safe:
                    present[t] = True
                    for ci in triple_to_configs[t]:
                        config_count[ci] += 1
                    improved = True
                    if time.time() > deadline - 5:
                        break
        return present, config_count

    def simulated_annealing(present, config_count):
        current_present = present[:]
        current_config_count = config_count[:]
        current_size = sum(current_present)
        best_present = current_present[:]
        best_config_count = current_config_count[:]
        best_size = current_size
        
        T = 5.0
        cooling_rate = 0.9995
        min_T = 0.1
        
        while T > min_T and time.time() < deadline - 10:
            move_type = random.random()
            if move_type < 0.4:
                t = random.randint(0, num_triples - 1)
                if not current_present[t]:
                    violations = 0
                    for ci in triple_to_configs[t]:
                        if current_config_count[ci] == 7:
                            violations += 1
                    if violations == 0:
                        current_present[t] = True
                        for ci in triple_to_configs[t]:
                            current_config_count[ci] += 1
                        current_size += 1
                        if current_size > best_size:
                            best_size = current_size
                            best_present = current_present[:]
                            best_config_count = current_config_count[:]
            elif move_type < 0.7:
                t = random.randint(0, num_triples - 1)
                if current_present[t]:
                    if random.random() < math.exp(1.0 / T):
                        current_present[t] = False
                        for ci in triple_to_configs[t]:
                            current_config_count[ci] -= 1
                        current_size -= 1
            else:
                t_out = random.randint(0, num_triples - 1)
                t_in = random.randint(0, num_triples - 1)
                if current_present[t_out] and not current_present[t_in] and t_out != t_in:
                    current_present[t_out] = False
                    for ci in triple_to_configs[t_out]:
                        current_config_count[ci] -= 1
                    safe = True
                    for ci in triple_to_configs[t_in]:
                        if current_config_count[ci] == 7:
                            safe = False
                            break
                    if safe:
                        current_present[t_in] = True
                        for ci in triple_to_configs[t_in]:
                            current_config_count[ci] += 1
                    else:
                        current_present[t_out] = True
                        for ci in triple_to_configs[t_out]:
                            current_config_count[ci] += 1
            T *= cooling_rate
        return best_present, best_config_count

    best_overall_present = []
    best_overall_size = 0
    
    while time.time() < deadline - 15:
        present, config_count = greedy_init()
        present, config_count = local_search(present, config_count)
        present, config_count = simulated_annealing(present, config_count)
        size = sum(present)
        if size > best_overall_size:
            best_overall_size = size
            best_overall_present = present[:]
            
    result_indices = [i for i in range(num_triples) if best_overall_present[i]]
    result_set = set(result_indices)
    
    violated_configs = []
    for ci, config in enumerate(forbidden_configs):
        if all(t in result_set for t in config):
            violated_configs.append(ci)
            
    if violated_configs:
        current_present = best_overall_present[:]
        current_config_count = [0] * num_configs
        for t in result_indices:
            for ci in triple_to_configs[t]:
                current_config_count[ci] += 1
                
        changed = True
        while changed and time.time() < deadline - 2:
            changed = False
            for ci in violated_configs:
                if current_config_count[ci] == 8:
                    best_t = -1
                    max_violations_fixed = 0
                    for t in config_to_triples[ci]:
                        if current_present[t]:
                            v_count = 0
                            for c2 in triple_to_configs[t]:
                                if c2 in violated_configs and current_config_count[c2] == 8:
                                    v_count += 1
                            if v_count > max_violations_fixed:
                                max_violations_fixed = v_count
                                best_t = t
                    if best_t != -1:
                        current_present[best_t] = False
                        for c2 in triple_to_configs[best_t]:
                            current_config_count[c2] -= 1
                        violated_configs = [c for c in range(num_configs) if current_config_count[c] == 8]
                        changed = True
                        break
            if not violated_configs:
                break
        best_overall_present = current_present

    return [list(all_triples[i]) for i in range(num_triples) if best_overall_present[i]]
