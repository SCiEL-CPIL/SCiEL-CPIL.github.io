"""
Genetic Algorithm with efficient local search to minimize monochromatic 4-term APs in [1, 256].
"""
import numpy as np
import time

N = 256

# Precompute all 4-term arithmetic progressions (a, a+d, a+2d, a+3d) with 1 <= a, a+3d <= 256, d >= 1
# Represent each AP as 0-based indices
def _precompute_aps():
    aps = []
    for d in range(1, N):
        for a in range(1, N - 3 * d + 1):
            # 1-based: a, a+d, a+2d, a+3d
            # 0-based: a-1, a+d-1, a+2d-1, a+3d-1
            aps.append((a - 1, a + d - 1, a + 2 * d - 1, a + 3 * d - 1))
    return np.array(aps, dtype=np.int32)

AP_IDX = _precompute_aps()
NUM_APS = AP_IDX.shape[0]

# Precompute for each position i, which APs contain i.
# Also precompute the "other 3 indices" for each (position, AP) pair.
AP_CONTAINING_POS = [list() for _ in range(N)]
OTHER_3_IDX = [list() for _ in range(N)]

for i in range(N):
    indices_in_aps = np.where(AP_IDX == i)
    ap_indices_list = indices_in_aps[0].tolist()
    AP_CONTAINING_POS[i] = ap_indices_list
    for k in ap_indices_list:
        idxs = AP_IDX[k]
        other = []
        for j in range(4):
            if idxs[j] != i:
                other.append(idxs[j])
        OTHER_3_IDX[i].append(tuple(other))

def _count_mono_aps(colours):
    """Count monochromatic 4-term APs. colours is a numpy array of 0/1 of length 256."""
    cols = colours[AP_IDX]
    mono = (cols[:, 0] == cols[:, 1]) & (cols[:, 1] == cols[:, 2]) & (cols[:, 2] == cols[:, 3])
    return int(np.sum(mono))

def _local_search(colours, max_sweeps=50, rng=None):
    """
    Fast local search using precomputed AP containment and delta evaluation.
    Returns (improved_colours, score).
    """
    if rng is None:
        rng = np.random.RandomState(0)
    
    colours = colours.copy()
    cols = colours[AP_IDX]
    mono_status = (cols[:, 0] == cols[:, 1]) & (cols[:, 1] == cols[:, 2]) & (cols[:, 2] == cols[:, 3])
    current_count = int(np.sum(mono_status))
    
    for sweep in range(max_sweeps):
        improved = False
        order = np.arange(N)
        rng.shuffle(order)
        
        for pos in order:
            old_val = colours[pos]
            new_val = 1 - old_val
            
            delta = 0
            affected_aps = AP_CONTAINING_POS[pos]
            other_3s = OTHER_3_IDX[pos]
            
            for j, k in enumerate(affected_aps):
                other3 = other_3s[j]
                v1 = colours[other3[0]]
                v2 = colours[other3[1]]
                v3 = colours[other3[2]]
                
                if v1 == new_val and v2 == new_val and v3 == new_val:
                    if not mono_status[k]:
                        delta += 1
                else:
                    if mono_status[k]:
                        delta -= 1
            
            if delta < 0:
                colours[pos] = new_val
                current_count += delta
                for j, k in enumerate(affected_aps):
                    other3 = other_3s[j]
                    v1 = colours[other3[0]]
                    v2 = colours[other3[1]]
                    v3 = colours[other3[2]]
                    mono_status[k] = (new_val == v1) and (v1 == v2) and (v2 == v3)
                improved = True
        
        if not improved:
            break
    
    return colours, current_count

def construct():
    """
    Genetic algorithm to minimize monochromatic 4-term APs in [1, 256].
    Returns 256 colours in {0, 1}.
    """
    start_time = time.time()
    time_limit = 115  # seconds

    rng = np.random.RandomState(42)
    
    pop_size = 40
    population = []
    scores = []
    
    for _ in range(pop_size // 4):
        c = rng.randint(0, 2, size=N)
        population.append(c)
        
    def thue_morse(n):
        tm = [0]
        while len(tm) < n:
            tm = tm + [1 - x for x in tm]
        return np.array(tm[:n], dtype=np.int8)
    population.append(thue_morse(N))
    
    for block_size in [2, 3, 4, 5, 6, 7, 8, 12, 16]:
        c = np.array([(i // block_size) % 2 for i in range(N)], dtype=np.int8)
        population.append(c)
        
    for period in [3, 4, 5, 6, 7, 8, 10, 12]:
        half = period // 2
        c = np.array([(i % period) < half for i in range(N)], dtype=np.int8)
        population.append(c)
    
    c = np.array([(i & 1) ^ ((i >> 1) & 1) for i in range(N)], dtype=np.int8)
    population.append(c)
    
    for _ in range(pop_size // 4):
        c = rng.randint(0, 2, size=N)
        population.append(c)
        
    while len(population) < pop_size:
        c = rng.randint(0, 2, size=N)
        population.append(c)
        
    for c in population:
        scores.append(_count_mono_aps(c))
        
    top_indices = np.argsort(scores)[:10]
    for idx in top_indices:
        improved_c, improved_s = _local_search(population[idx], max_sweeps=20, rng=rng)
        population[idx] = improved_c
        scores[idx] = improved_s
    
    best_idx = np.argmin(scores)
    best_score = scores[best_idx]
    best_individual = population[best_idx].copy()
    
    generation = 0
    while time.time() - start_time < time_limit:
        generation += 1
        
        order = np.argsort(scores)
        population = [population[i] for i in order]
        scores = [scores[i] for i in order]
        
        elite_size = 5
        new_population = [population[i].copy() for i in range(elite_size)]
        new_scores = scores[:elite_size]
        
        while len(new_population) < pop_size:
            def tournament_select():
                candidates = rng.randint(0, len(population), size=3)
                best_c = min(candidates, key=lambda i: scores[i])
                return population[best_c]
            
            p1 = tournament_select()
            p2 = tournament_select()
            
            mask = rng.randint(0, 2, size=N)
            child = np.where(mask == 1, p1, p2).copy()
            
            num_mut = rng.randint(2, 8)
            for _ in range(num_mut):
                pos = rng.randint(0, N)
                child[pos] = 1 - child[pos]
            
            child, child_score = _local_search(child, max_sweeps=30, rng=rng)
            
            new_population.append(child)
            new_scores.append(child_score)
            
        population = new_population
        scores = new_scores
        
        best_idx = np.argmin(scores)
        if scores[best_idx] < best_score:
            best_score = scores[best_idx]
            best_individual = population[best_idx].copy()
        
        if generation % 10 == 0 and time.time() - start_time < time_limit - 10:
            for _ in range(3):
                c = rng.randint(0, 2, size=N)
                c, s = _local_search(c, max_sweeps=15, rng=rng)
                worst_idx = np.argmax(scores)
                population[worst_idx] = c
                scores[worst_idx] = s
                if s < best_score:
                    best_score = s
                    best_individual = c.copy()
            
            population = [population[i] for i in np.argsort(scores)]
            scores = [scores[i] for i in np.argsort(scores)]
            
    return [int(x) for x in best_individual]

if __name__ == "__main__":
    result = construct()
    assert len(result) == 256
    assert all(x in (0, 1) for x in result)
    count = _count_mono_aps(np.array(result, dtype=np.int8))
    print(f"Monochromatic 4-APs: {count}")
