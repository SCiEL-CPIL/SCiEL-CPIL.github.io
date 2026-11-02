import time
import random
import math
from itertools import permutations
from collections import defaultdict

def construct(n):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 150s limit
    
    if n == 1:
        return "1"
    
    # Standard recursive construction (Graham-Lagarias)
    # This produces a valid superpermutation of length 1! + 2! + ... + n!
    word = [1]
    for m in range(2, n + 1):
        perms, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        new_word = []
        for p in perms:
            block = list(p) + [m] + list(p)
            max_overlap = min(len(new_word), len(block))
            overlap = 0
            for o in range(max_overlap, -1, -1):
                if new_word[len(new_word) - o:] == block[:o]:
                    overlap = o
                    break
            new_word += block[overlap:]
        word = new_word
    
    if n <= 6:
        return "".join(map(str, word))
    
    if n == 7:
        # Use a graph-based approach: find a Hamiltonian path on S_6 that maximizes overlap
        perms_6 = list(permutations(range(1, 7)))
        num_p = len(perms_6)
        
        # Build overlap matrix: overlap[i][j] = max k such that perms_6[i][-k:] == perms_6[j][:k]
        overlap = [[0]*num_p for _ in range(num_p)]
        for i in range(num_p):
            pi = perms_6[i]
            for j in range(num_p):
                if i == j:
                    continue
                pj = perms_6[j]
                # Check k from 5 down to 0
                for k in range(5, -1, -1):
                    if pi[-k:] == pj[:k]:
                        overlap[i][j] = k
                        break
        
        # Greedy construction: start with random node, always go to unvisited node with max overlap
        def greedy_construct(start_node, random_seed=None):
            if random_seed is not None:
                random.seed(random_seed)
            
            visited = [False] * num_p
            path = [start_node]
            visited[start_node] = True
            current = start_node
            
            for _ in range(num_p - 1):
                best_next = -1
                best_ov = -1
                # Collect all unvisited neighbors
                candidates = []
                for j in range(num_p):
                    if not visited[j] and overlap[current][j] > 0:
                        candidates.append((overlap[current][j], j))
                
                if not candidates:
                    # No positive overlap, pick random unvisited
                    unvisited = [j for j in range(num_p) if not visited[j]]
                    if not unvisited:
                        break
                    j = random.choice(unvisited)
                else:
                    # Pick the one with max overlap, break ties randomly
                    candidates.sort(reverse=True)
                    max_ov = candidates[0][0]
                    top = [c for c in candidates if c[0] == max_ov]
                    j = top[random.randint(0, len(top)-1)][1]
                
                path.append(j)
                visited[j] = True
                current = j
            
            return path
        
        # Try multiple greedy constructions with different starts and seeds
        best_path = None
        best_overlap_sum = -1
        
        # Try several random starts
        for seed in range(20):
            start_node = random.randint(0, num_p - 1)
            path = greedy_construct(start_node, seed)
            ov_sum = sum(overlap[path[i]][path[i+1]] for i in range(len(path)-1))
            if ov_sum > best_overlap_sum:
                best_overlap_sum = ov_sum
                best_path = path
        
        # Local search: 2-opt on the path
        def path_overlap_sum(path):
            return sum(overlap[path[i]][path[i+1]] for i in range(len(path)-1))
        
        current_path = best_path[:]
        current_ov = best_overlap_sum
        improved = True
        iterations = 0
        while improved and time.time() < deadline:
            improved = False
            iterations += 1
            for i in range(num_p - 1):
                if time.time() > deadline:
                    break
                for j in range(i+1, num_p):
                    # Reverse segment i..j
                    new_path = current_path[:i] + current_path[i:j+1][::-1] + current_path[j+1:]
                    new_ov = path_overlap_sum(new_path)
                    if new_ov > current_ov:
                        current_path = new_path
                        current_ov = new_ov
                        improved = True
                        break
                if improved:
                    break
        
        # Construct word from optimized path
        new_word = []
        for idx in current_path:
            p = perms_6[idx]
            block = list(p) + [7] + list(p)
            max_ov = min(len(new_word), len(block))
            ov = 0
            for o in range(max_ov, -1, -1):
                if new_word[len(new_word) - o:] == block[:o]:
                    ov = o
                    break
            new_word += block[ov:]
        
        # Verify validity
        if len(new_word) >= 7:
            seen = set()
            for i in range(len(new_word) - 6):
                w = tuple(new_word[i:i+7])
                if len(set(w)) == 7:
                    seen.add(w)
            if len(seen) == 5040 and len(new_word) < len(word):
                word = new_word
    
    return "".join(map(str, word))
