"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _time_left(deadline, margin=1.0):
    return (deadline - time.time()) > margin

def _has_k4(adj, n):
    # adj: list of int bitsets; check for K4
    for i in range(n):
        for j in range(i + 1, n):
            if not (adj[i] >> j) & 1:
                continue
            # common neighbors of i and j
            common = adj[i] & adj[j]
            # remove i and j
            common &= ~(1 << i)
            common &= ~(1 << j)
            # check if any edge among common neighbors
            bits = []
            x = common
            while x:
                lsb = x & (-x)
                idx = lsb.bit_length() - 1
                bits.append(idx)
                x ^= lsb
            for a_idx in range(len(bits)):
                for b_idx in range(a_idx + 1, len(bits)):
                    a = bits[a_idx]
                    b = bits[b_idx]
                    if (adj[a] >> b) & 1:
                        return True
    return False

def _has_indep6(adj, n):
    # independent set of size 6: in complement, clique of size 6
    # complement adj
    comp = [0] * n
    for i in range(n):
        comp[i] = ((1 << n) - 1) ^ (1 << i) ^ adj[i]
    # find K6 in comp
    for i in range(n):
        for j in range(i + 1, n):
            if not (comp[i] >> j) & 1:
                continue
            common = comp[i] & comp[j]
            common &= ~(1 << i)
            common &= ~(1 << j)
            bits = []
            x = common
            while x:
                lsb = x & (-x)
                idx = lsb.bit_length() - 1
                bits.append(idx)
                x ^= lsb
            for a_idx in range(len(bits)):
                for b_idx in range(a_idx + 1, len(bits)):
                    for c_idx in range(b_idx + 1, len(bits)):
                        a = bits[a_idx]
                        b = bits[b_idx]
                        c = bits[c_idx]
                        if not ((comp[a] >> b) & 1 and (comp[a] >> c) & 1 and (comp[b] >> c) & 1):
                            continue
                        # check fourth vertex
                        common2 = comp[a] & comp[b] & comp[c]
                        common2 &= ~(1 << i)
                        common2 &= ~(1 << j)
                        common2 &= ~(1 << a)
                        common2 &= ~(1 << b)
                        common2 &= ~(1 << c)
                        if common2:
                            return True
    return False

def _is_valid(adj, n):
    return not _has_k4(adj, n) and not _has_indep6(adj, n)

def _adj_to_edges(adj, n):
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if (adj[i] >> j) & 1:
                edges.append([i, j])
    return edges

def _random_adj(n, p=0.5):
    adj = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < p:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    return adj

def _local_search(adj, n, deadline):
    # try to repair violations by flipping edges
    best_adj = adj[:]
    best_n = n
    while _time_left(deadline, 2.0):
        if _is_valid(best_adj, best_n):
            return best_adj, best_n
        # find a K4 or indep6 to break
        # Try random edge flips
        for _ in range(50):
            if not _time_left(deadline, 2.0):
                break
            i = random.randint(0, best_n - 1)
            j = random.randint(0, best_n - 1)
            if i == j:
                continue
            if i > j:
                i, j = j, i
            # flip edge
            old = (best_adj[i] >> j) & 1
            if old:
                best_adj[i] &= ~(1 << j)
                best_adj[j] &= ~(1 << i)
            else:
                best_adj[i] |= (1 << j)
                best_adj[j] |= (1 << i)
            if _is_valid(best_adj, best_n):
                return best_adj, best_n
            # revert
            if old:
                best_adj[i] |= (1 << j)
                best_adj[j] |= (1 << i)
            else:
                best_adj[i] &= ~(1 << j)
                best_adj[j] &= ~(1 << i)
    return best_adj, best_n

def construct():
    deadline = time.time() + 140.0  # 10s safety margin
    best_n = 1
    best_adj = None
    
    # Start from small n and try to grow
    for target_n in range(3, 37):
        if not _time_left(deadline, 5.0):
            break
        found = False
        attempts = 0
        while not found and attempts < 200 and _time_left(deadline, 5.0):
            attempts += 1
            # random graph with bias toward no K4 and no indep6
            # For larger n, need careful density
            p = 0.4 if target_n <= 20 else 0.3
            adj = _random_adj(target_n, p)
            adj, n = _local_search(adj, target_n, deadline)
            if n >= target_n and _is_valid(adj, target_n):
                best_n = target_n
                best_adj = adj
                found = True
                break
    
    if best_adj is None:
        # fallback: Paley 17
        n = 17
        adj = [0] * n
        squares = {x * x % 17 for x in range(1, 17)}
        for i in range(n):
            for j in range(i + 1, n):
                if (j - i) % 17 in squares:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        best_n = n
        best_adj = adj
    
    edges = _adj_to_edges(best_adj, best_n)
    return {'n': best_n, 'edges': edges}
# EVOLVE-BLOCK-END
