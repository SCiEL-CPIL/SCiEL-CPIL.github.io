import time
import sys
from math import gcd
import random

def construct():
    start = time.time()
    deadline = start + 85.0  # Safety margin before 90s kill

    def elapsed():
        return time.time() - start

    N = 256
    numbers = list(range(1, N + 1))

    # Conflict check: (a+b) | (a*b)
    # Let g = gcd(a,b), a=gu, b=gv, gcd(u,v)=1
    # Condition: (u+v) | g
    def inc(a, b):
        if a == b:
            return False
        g = gcd(a, b)
        u = a // g
        v = b // g
        return (u + v) % g == 0

    # Build full conflict graph
    n = N
    # adj[i] is a bitmask of j such that numbers[i] and numbers[j] conflict
    adj = [0] * n
    for i in range(n):
        ai = numbers[i]
        mask = 0
        for j in range(n):
            if i == j:
                continue
            aj = numbers[j]
            if inc(ai, aj):
                mask |= (1 << j)
        adj[i] = mask

    def popcount(x):
        return bin(x).count('1')

    # Min-degree greedy independent set
    def greedy_min_degree(seed=None):
        if seed is not None:
            random.seed(seed)
        
        available = (1 << n) - 1
        selected = []
        
        # Precompute degrees
        deg = [popcount(adj[i]) for i in range(n)]
        
        while available:
            if elapsed() > deadline:
                break
            
            # Find vertex with minimum degree in available subgraph
            min_deg = n + 1
            min_v = -1
            temp = available
            while temp:
                v = (temp & (-temp)).bit_length() - 1
                d = popcount(adj[v] & available)
                if d < min_deg:
                    min_deg = d
                    min_v = v
                    if d == 0:
                        break
                temp &= temp - 1
            
            if min_v < 0:
                break
            
            selected.append(min_v)
            
            # Remove min_v and its neighbors from available
            available &= ~(1 << min_v)
            available &= ~adj[min_v]
        
        return selected

    # Local search: try to add vertices or swap
    def local_search(selected, max_iter=1000):
        selected_set = set(selected)
        selected_list = list(selected)
        
        for _ in range(max_iter):
            if elapsed() > deadline:
                break
            
            improved = False
            
            # Try adding a vertex not in selected_set
            for i in range(n):
                if i in selected_set:
                    continue
                # Check if i conflicts with any in selected_set
                conflict = False
                for j in selected_set:
                    if adj[i] & (1 << j):
                        conflict = True
                        break
                if not conflict:
                    selected_set.add(i)
                    selected_list.append(i)
                    improved = True
                    break
            
            if improved:
                continue
            
            # Try swapping: remove one, add another
            for i in range(n):
                if i in selected_set:
                    continue
                # Find a vertex j in selected_set such that removing j allows adding i
                # and no other conflicts
                for j in list(selected_set):
                    if j == i:
                        continue
                    # Check if i conflicts with any in selected_set except j
                    conflict = False
                    for k in selected_set:
                        if k == j:
                            continue
                        if adj[i] & (1 << k):
                            conflict = True
                            break
                    if not conflict:
                        selected_set.remove(j)
                        selected_set.add(i)
                        # Update selected_list
                        selected_list.remove(j)
                        selected_list.append(i)
                        improved = True
                        break
                if improved:
                    break
            
            if not improved:
                break
        
        return selected_list

    # Run multiple randomized trials
    best_size = 0
    best_set = []
    
    # First, run a few deterministic greedy passes
    for seed in [0, 1, 2, 3, 4]:
        if elapsed() > deadline - 10:
            break
        
        trial = greedy_min_degree(seed=seed)
        if len(trial) > best_size:
            best_size = len(trial)
            best_set = trial[:]
        
        # Local search refinement
        refined = local_search(trial, max_iter=500)
        if len(refined) > best_size:
            best_size = len(refined)
            best_set = refined[:]
    
    # More randomized trials if time permits
    trial_count = 5
    while elapsed() < deadline - 15:
        if elapsed() > deadline - 5:
            break
        
        seed = random.randint(0, 10**6)
        trial = greedy_min_degree(seed=seed)
        if len(trial) > best_size:
            best_size = len(trial)
            best_set = trial[:]
        
        # Local search refinement
        refined = local_search(trial, max_iter=300)
        if len(refined) > best_size:
            best_size = len(refined)
            best_set = refined[:]
        
        trial_count += 1
        if trial_count > 50:
            break

    # Convert indices back to values
    final_set = set(numbers[i] for i in best_set)
    
    # Verification
    for a in final_set:
        for b in final_set:
            if a != b and (a * b) % (a + b) == 0:
                # Invalid, fall back to a simple valid set
                return [1]
    
    return sorted(final_set)

if __name__ == "__main__":
    result = construct()
    print(len(result))
