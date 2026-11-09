import time
import itertools
import random

_START = time.time()
_LIMIT = 115.0  # Safety margin under 120s (or 150s? Prompt says 150s kill, 120s finish. Let's use 115 to be safe).

def _deadline_ok():
    return (time.time() - _START) < _LIMIT

def _random_edges(n, p, seed):
    from random import Random
    rng = Random(seed)
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                edges.append((u, v))
    return edges

def _edge_key(u, v):
    return (u, v) if u < v else (v, u)

def _adj_mask(n, edges):
    m = [0] * n
    for (u, v) in edges:
        m[u] |= 1 << v
        m[v] |= 1 << u
    return m

def _iter_bits(x):
    while x:
        lsb = x & -x
        yield lsb.bit_length() - 1
        x ^= lsb

def _maximal_cliques_bit(adj_mask, n, min_size=3, cap=300000, deadline_ok=None):
    result = []
    aborted = [False]

    def bronk(R, P, X):
        if aborted[0]:
            return
        if deadline_ok is not None and not deadline_ok():
            aborted[0] = True
            return
        if not P and not X:
            if R.bit_count() >= min_size:
                result.append(R)
                if len(result) >= cap:
                    aborted[0] = True
            return
        if not P:
            return
        ux = P | X
        best_u = -1
        best_c = -1
        for u in _iter_bits(ux):
            c = (P & adj_mask[u]).bit_count()
            if c > best_c:
                best_c = c
                best_u = u
        for v in _iter_bits(P & ~adj_mask[best_u]):
            bit = 1 << v
            bronk(R | bit, P & adj_mask[v], X & adj_mask[v])
            if aborted[0]:
                return
            P &= ~bit
            X |= bit

    bronk(0, (1 << n) - 1, 0)
    return result

def _mask_to_list(mask, n):
    return [v for v in range(n) if (mask >> v) & 1]

def _clique_edges(clique):
    return frozenset(_edge_key(u, v) for u, v in itertools.combinations(clique, 2))

def _greedy_clique_bit(adj_mask, n):
    order = sorted(range(n), key=lambda x: -(adj_mask[x].bit_count()))
    cand = (1 << n) - 1
    clique = []
    for v in order:
        if not ((cand >> v) & 1):
            continue
        clique.append(v)
        cand &= adj_mask[v]
        if not cand:
            break
    return clique

def _solve_instance(n, p, seed, rng_seed=None, top_k=1):
    edges = _random_edges(n, p, seed)
    all_edges = set(_edge_key(u, v) for (u, v) in edges)
    chosen = []
    used = set()
    remaining = set(all_edges)
    
    # Use a separate RNG for decisions
    if rng_seed is not None:
        rng = random.Random(rng_seed)
    else:
        rng = None

    round_i = 0
    while remaining and _deadline_ok() and round_i < 50:
        round_i += 1
        rem_mask = _adj_mask(n, remaining)
        maxcls = _maximal_cliques_bit(rem_mask, n, 3, 300000, _deadline_ok)
        candidates = [_mask_to_list(m, n) for m in maxcls]
        
        if not candidates:
            c = _greedy_clique_bit(rem_mask, n)
            if len(c) >= 3:
                candidates = [c]
            else:
                break
        
        # Sort by size descending
        candidates.sort(key=lambda c: -len(c))
        
        # Selection strategy
        selected_clique = None
        if top_k > 1 and rng is not None and len(candidates) > 1:
            # Pick from top K, but ensure we don't pick a significantly smaller one if a big one exists
            # Let's pick from the top min(top_k, len(candidates))
            k = min(top_k, len(candidates))
            # Weighted random? Or uniform?
            # Uniform among top K is a good exploration strategy.
            selected_clique = rng.choice(candidates[:k])
        else:
            selected_clique = candidates[0]
            
        if selected_clique:
            ec = _clique_edges(selected_clique)
            # Check if valid (all edges remaining)
            if ec <= remaining:
                used |= ec
                remaining -= ec
                chosen.append(selected_clique)
            else:
                # Should not happen if we computed maxcls on remaining, but safety
                break
        else:
            break

    for e in remaining:
        chosen.append([e[0], e[1]])
    return chosen

def construct():
    # Instance 1
    n1, p1, s1 = 32, 0.6, 601
    # Instance 2
    n2, p2, s2 = 64, 0.55, 602
    
    best_res1 = None
    best_res2 = None
    best_score = float('inf')
    
    # Strategy: 
    # 1. Deterministic run
    # 2. Several randomized runs with increasing top_k
    
    runs = [
        (0, 1),   # Deterministic
        (101, 5), # Random, top 5
        (102, 10),# Random, top 10
        (103, 20),# Random, top 20
        (104, 50),# Random, top 50
    ]
    
    for idx, (rng_seed, top_k) in enumerate(runs):
        if not _deadline_ok():
            break
            
        # Solve instance 1
        res1 = _solve_instance(n1, p1, s1, rng_seed, top_k)
        if not _deadline_ok():
            break
            
        # Solve instance 2
        res2 = _solve_instance(n2, p2, s2, rng_seed, top_k)
        
        score = len(res1) + len(res2)
        if score < best_score:
            best_score = score
            best_res1 = res1
            best_res2 = res2
            
        # If we have a very good score, we might stop early? 
        # Lower bound: number of edges / max clique edges.
        # For n=64, p=0.55, E ~ 1000. Max clique ~ 6 (15 edges). LB ~ 67.
        # If we get close to LB, we can stop.
        
    if best_res1 is None:
        # Fallback to deterministic if all failed
        best_res1 = _solve_instance(n1, p1, s1, 0, 1)
        best_res2 = _solve_instance(n2, p2, s2, 0, 1)

    return [best_res1, best_res2]
