import itertools
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 80.0
    
    n = 12
    m = 40
    all_elements = list(range(n))
    
    # Generate all subsets
    all_subsets = []
    for k in range(n + 1):
        for combo in itertools.combinations(all_elements, k):
            all_subsets.append(frozenset(combo))
    
    fs_to_idx = {fs: i for i, fs in enumerate(all_subsets)}
    N = len(all_subsets)
    
    # Precompute comparability
    comparable = [set() for _ in range(N)]
    for i in range(N):
        si = all_subsets[i]
        for j in range(N):
            if i != j:
                sj = all_subsets[j]
                if si < sj or sj < si:
                    comparable[i].add(j)
                    comparable[j].add(i)
    
    # Initial: chain
    chain = [frozenset(range(k)) for k in range(n + 1)]
    current = list(chain)
    current_set = set(chain)
    
    available = [i for i in range(N) if all_subsets[i] not in current_set]
    
    def compute_score(fam):
        count = 0
        for i in range(len(fam)):
            for j in range(i + 1, len(fam)):
                if fam[i] < fam[j] or fam[j] < fam[i]:
                    count += 1
        return count
    
    best_score = compute_score(current)
    best_family = list(current)
    
    # Greedy add
    while len(current) < m and time.time() < deadline - 40:
        best_gain = -1
        best_idx = -1
        for idx in available:
            s = all_subsets[idx]
            gain = sum(1 for c in current if s < c or c < s)
            if gain > best_gain:
                best_gain = gain
                best_idx = idx
        if best_idx == -1:
            break
        s = all_subsets[best_idx]
        current.append(s)
        current_set.add(s)
        available.remove(best_idx)
        best_score += best_gain
    
    # Local search
    random.seed(42)
    for iteration in range(1000):
        if time.time() > deadline - 10:
            break
        
        improved = False
        for _ in range(3000):
            if time.time() > deadline - 10:
                break
            i = random.randint(0, m - 1)
            old_set = current[i]
            old_idx = fs_to_idx[old_set]
            
            if not available:
                break
            new_idx = random.choice(available)
            new_set = all_subsets[new_idx]
            
            others = [current[k] for k in range(m) if k != i]
            old_comp = sum(1 for o in others if old_set < o or o < old_set)
            new_comp = sum(1 for o in others if new_set < o or o < new_set)
            
            if new_comp > old_comp:
                current[i] = new_set
                current_set.remove(old_set)
                current_set.add(new_set)
                available.remove(new_idx)
                available.append(old_idx)
                best_score += (new_comp - old_comp)
                improved = True
                if best_score > best_score:
                    best_family = list(current)
        
        if not improved:
            for _ in range(5):
                if time.time() > deadline - 10:
                    break
                i = random.randint(0, m - 1)
                if not available:
                    break
                new_idx = random.choice(available)
                old_set = current[i]
                old_idx = fs_to_idx[old_set]
                current[i] = all_subsets[new_idx]
                current_set.remove(old_set)
                current_set.add(all_subsets[new_idx])
                available.remove(new_idx)
                available.append(old_idx)
    
    final = best_family if best_score >= compute_score(current) else current
    return [sorted(list(s)) for s in final]
