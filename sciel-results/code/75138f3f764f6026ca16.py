import itertools
import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s margin
    
    def get_fallback():
        n = 29
        p = 29
        colors = []
        for a, b, c in itertools.combinations(range(n), 3):
            if (a**3 + b**3 + c**3) % p == 0:
                colors.append(0)
            else:
                colors.append(1)
        return {'n': n, 'colors': colors}

    N = 34
    triples = list(itertools.combinations(range(N), 3))
    num_triples = len(triples)
    triple_index = {t: i for i, t in enumerate(triples)}
    
    k4s = list(itertools.combinations(range(N), 4))
    k5s = list(itertools.combinations(range(N), 5))
    num_k4 = len(k4s)
    num_k5 = len(k5s)
    
    triple_to_k4s = [[] for _ in range(num_triples)]
    triple_to_k5s = [[] for _ in range(num_triples)]
    
    red_counts = [0] * num_k4
    blue_counts = [0] * num_k5
    
    for k4_idx, k4 in enumerate(k4s):
        t1 = (k4[0], k4[1], k4[2])
        t2 = (k4[0], k4[1], k4[3])
        t3 = (k4[0], k4[2], k4[3])
        t4 = (k4[1], k4[2], k4[3])
        
        i1 = triple_index[t1]
        i2 = triple_index[t2]
        i3 = triple_index[t3]
        i4 = triple_index[t4]
        
        triple_to_k4s[i1].append(k4_idx)
        triple_to_k4s[i2].append(k4_idx)
        triple_to_k4s[i3].append(k4_idx)
        triple_to_k4s[i4].append(k4_idx)
        
    for k5_idx, k5 in enumerate(k5s):
        for t in itertools.combinations(k5, 3):
            i = triple_index[t]
            triple_to_k5s[i].append(k5_idx)
            
    colors = [1] * num_triples
    
    blue_counts = [10] * num_k5
    red_counts = [0] * num_k4
    
    violating_k4s = set()
    violating_k5s = set(range(num_k5))
    
    random.seed(42)
    
    while time.time() < deadline:
        if not violating_k4s and not violating_k5s:
            return {'n': N, 'colors': colors}
        
        fix_red = False
        if violating_k4s:
            if not violating_k5s or random.random() < 0.5:
                fix_red = True
        
        if fix_red:
            k4_idx = random.choice(list(violating_k4s))
            k4 = k4s[k4_idx]
            edges_in_k4 = [
                (k4[0], k4[1], k4[2]),
                (k4[0], k4[1], k4[3]),
                (k4[0], k4[2], k4[3]),
                (k4[1], k4[2], k4[3])
            ]
            edge = random.choice(edges_in_k4)
            edge_idx = triple_index[edge]
            
            if colors[edge_idx] == 0:
                colors[edge_idx] = 1
                
                for k4_i in triple_to_k4s[edge_idx]:
                    red_counts[k4_i] -= 1
                    if red_counts[k4_i] == 4:
                        violating_k4s.add(k4_i)
                    elif red_counts[k4_i] == 3 and k4_i in violating_k4s:
                        violating_k4s.remove(k4_i)
                
                for k5_i in triple_to_k5s[edge_idx]:
                    blue_counts[k5_i] += 1
                    if blue_counts[k5_i] == 10:
                        violating_k5s.add(k5_i)
                    elif blue_counts[k5_i] == 9 and k5_i in violating_k5s:
                        violating_k5s.remove(k5_i)
        else:
            k5_idx = random.choice(list(violating_k5s))
            k5 = k5s[k5_idx]
            edges_in_k5 = list(itertools.combinations(k5, 3))
            edge = random.choice(edges_in_k5)
            edge_idx = triple_index[edge]
            
            if colors[edge_idx] == 1:
                colors[edge_idx] = 0
                
                for k4_i in triple_to_k4s[edge_idx]:
                    red_counts[k4_i] += 1
                    if red_counts[k4_i] == 4:
                        violating_k4s.add(k4_i)
                    elif red_counts[k4_i] == 3 and k4_i in violating_k4s:
                        violating_k4s.remove(k4_i)
                
                for k5_i in triple_to_k5s[edge_idx]:
                    blue_counts[k5_i] -= 1
                    if blue_counts[k5_i] == 10:
                        violating_k5s.add(k5_i)
                    elif blue_counts[k5_i] == 9 and k5_i in violating_k5s:
                        violating_k5s.remove(k5_i)
    
    return get_fallback()
