import time
import math
import numpy as np

def construct() -> dict:
    start_time = time.time()
    deadline = start_time + 135.0

    def is_prime(n):
        if n < 2:
            return False
        if n < 4:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        i = 5
        while i * i <= n:
            if n % i == 0 or n % (i + 2) == 0:
                return False
            i += 6
        return True

    def make_paley_adjacency(p):
        residues = set()
        for x in range(1, p):
            residues.add((x * x) % p)
        adjacency = []
        for i in range(p):
            for j in range(i + 1, p):
                if (j - i) % p in residues:
                    adjacency.append(0)
                else:
                    adjacency.append(1)
        return adjacency

    def build_adjacency_matrix(p, adjacency):
        adj = np.zeros((p, p), dtype=bool)
        idx = 0
        for i in range(p):
            for j in range(i + 1, p):
                if adjacency[idx] == 0:
                    adj[i, j] = True
                else:
                    adj[j, i] = True
                idx += 1
        return adj

    def check_property(p, adjacency, time_limit):
        adj = build_adjacency_matrix(p, adjacency)
        in_neighbors = adj.T
        
        for u in range(p):
            if time.time() > time_limit:
                return None
            for v in range(u + 1, p):
                if time.time() > time_limit:
                    return None
                for w in range(v + 1, p):
                    if time.time() > time_limit:
                        return None
                    inter_3 = in_neighbors[u] & in_neighbors[v] & in_neighbors[w]
                    if not inter_3.any():
                        return False
                    for x in range(w + 1, p):
                        if time.time() > time_limit:
                            return None
                        if not (inter_3 & in_neighbors[x]).any():
                            return False
        return True

    primes_to_try = []
    for p in range(31, 70):
        if is_prime(p) and p % 4 == 3:
            primes_to_try.append(p)
    primes_to_try.sort()
    
    for p in primes_to_try:
        if time.time() > deadline - 15.0:
            break
        adj_list = make_paley_adjacency(p)
        result = check_property(p, adj_list, deadline - 5.0)
        if result is True:
            return {'n': p, 'orientation': adj_list}
        elif result is False:
            continue
        else:
            break
    
    adj_67 = make_paley_adjacency(67)
    return {'n': 67, 'orientation': adj_67}
