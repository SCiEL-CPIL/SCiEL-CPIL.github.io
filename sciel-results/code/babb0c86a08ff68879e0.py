import time
import random
import numpy as np

_START = time.time()
_DEADLINE = _START + 140.0  # Safety margin before 150s kill

# ---------------- Vectorized Verification Helpers ----------------

def _count_K4(n, bits):
    """
    Count the number of K4s in the graph.
    bits: list of ints, bits[i] is the adjacency bitset for vertex i.
    """
    count = 0
    # Iterate over pairs (a, b)
    for a in range(n):
        for b in range(a + 1, n):
            if not (bits[a] >> b) & 1:
                continue
            # Common neighbors of a and b, strictly greater than b
            common = bits[a] & bits[b]
            common &= ~((1 << (b + 1)) - 1)
            # Iterate over c in common
            # We need to count pairs (c, d) in common such that c and d are adjacent.
            # This is equivalent to counting edges in the subgraph induced by 'common'.
            # Let's just iterate c and check if there's any d > c in common that is adjacent to c.
            # Actually, for penalty, we just need to know if ANY exist, but counting helps gradient.
            # A faster way: for each c in common, check if (bits[c] & common & ~(1 << c)) has any bits.
            # But we want to count distinct K4s.
            # A K4 is {a,b,c,d}. a<b<c<d.
            # So we fix a,b. We need to find c,d in common(a,b) with c<d and edge(c,d).
            # Let's iterate c in common.
            c_list = [c for c in range(b + 1, n) if (common >> c) & 1]
            for i_c in range(len(c_list)):
                c = c_list[i_c]
                # d must be in common, d > c, and edge(c,d)
                d_mask = common & bits[c] & ~((1 << (c + 1)) - 1)
                # Count bits in d_mask
                count += bin(d_mask).count('1')
    return count

def _count_ind_6(n, bits):
    """
    Count the number of independent sets of size 6.
    """
    count = 0
    all_mask = (1 << n) - 1
    # Non-adjacency bitsets
    non_adj = [(all_mask ^ bits[i]) & ~(1 << i) for i in range(n)]
    
    for a in range(n):
        for b in range(a + 1, n):
            if (non_adj[a] >> b) & 1:
                common = non_adj[a] & non_adj[b]
                common &= ~((1 << (b + 1)) - 1)
                c_list = [c for c in range(b + 1, n) if (common >> c) & 1]
                for i_c in range(len(c_list)):
                    c = c_list[i_c]
                    common2 = common & non_adj[c]
                    common2 &= ~((1 << (c + 1)) - 1)
                    d_list = [d for d in range(c + 1, n) if (common2 >> d) & 1]
                    for i_d in range(len(d_list)):
                        d = d_list[i_d]
                        common3 = common2 & non_adj[d]
                        common3 &= ~((1 << (d + 1)) - 1)
                        e_list = [e for e in range(d + 1, n) if (common3 >> e) & 1]
                        for i_e in range(len(e_list)):
                            e = e_list[i_e]
                            common4 = common3 & non_adj[e]
                            common4 &= ~((1 << (e + 1)) - 1)
                            # Count bits in common4
                            count += bin(common4).count('1')
    return count

def _is_valid(n, bits):
    if _count_K4(n, bits) > 0:
        return False
    if _count_ind_6(n, bits) > 0:
        return False
    return True

def _edges_from_bits(n, bits):
    out = []
    for i in range(n):
        for j in range(i + 1, n):
            if (bits[i] >> j) & 1:
                out.append([i, j])
    return out

# ---------------- Paley Graph Construction ----------------

def _paley_graph(p):
    """Paley graph on p vertices where p is a prime congruent to 1 mod 4."""
    if p < 2 or p % 4 != 1:
        return None
    squares = {x * x % p for x in range(1, p)}
    bits = [0] * p
    for i in range(p):
        for j in range(i + 1, p):
            if (j - i) % p in squares:
                bits[i] |= (1 << j)
                bits[j] |= (1 << i)
    return bits

# ---------------- Genetic Algorithm ----------------

def _ga_search(n, pop_size=50, generations=100, time_limit=10.0, seed_bits=None):
    """
    Run GA to find a valid graph on n vertices.
    """
    start_time = time.time()
    rng = random.Random(42)
    
    # Initialize population
    population = []
    if seed_bits:
        population.append(seed_bits.copy())
    
    while len(population) < pop_size:
        bits = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if rng.random() < 0.5:
                    bits[i] |= (1 << j)
                    bits[j] |= (1 << i)
        population.append(bits)
    
    best_score = float('inf')
    best_bits = population[0].copy()
    
    for gen in range(generations):
        if time.time() - start_time > time_limit:
            break
        
        # Evaluate population
        scored_pop = []
        for bits in population:
            k4s = _count_K4(n, bits)
            ind6s = _count_ind_6(n, bits)
            # Penalty: We want to minimize (k4s, ind6s)
            # Use a weighted sum or lexicographic. Let's use sum + large weight for any violation.
            score = k4s + ind6s
            # If valid, score is 0.
            # To encourage valid graphs, we can add a small penalty for density deviation? No, just minimize violations.
            scored_pop.append((score, bits))
        
        scored_pop.sort(key=lambda x: x[0])
        
        if scored_pop[0][0] < best_score:
            best_score = scored_pop[0][0]
            best_bits = scored_pop[0][1].copy()
            
        if best_score == 0:
            return best_bits
            
        # Selection: Keep top half
        survivors = [b for _, b in scored_pop[:pop_size // 2]]
        
        # Create next generation
        next_pop = survivors.copy()
        
        while len(next_pop) < pop_size:
            # Crossover
            p1 = rng.choice(survivors)
            p2 = rng.choice(survivors)
            
            # Uniform crossover
            child = [0] * n
            for i in range(n):
                for j in range(i + 1, n):
                    if rng.random() < 0.5:
                        bit_val = (p1[i] >> j) & 1
                    else:
                        bit_val = (p2[i] >> j) & 1
                    if bit_val:
                        child[i] |= (1 << j)
                        child[j] |= (1 << i)
            
            # Mutation
            # Flip a few edges
            for _ in range(3):
                u = rng.randint(0, n - 1)
                v = rng.randint(0, n - 1)
                if u == v:
                    continue
                if u > v:
                    u, v = v, u
                if (child[u] >> v) & 1:
                    child[u] &= ~(1 << v)
                    child[v] &= ~(1 << u)
                else:
                    child[u] |= (1 << v)
                    child[v] |= (1 << u)
            
            next_pop.append(child)
        
        population = next_pop
        
    return best_bits

def _search_structured():
    best = {'n': 0, 'edges': []}
    
    # 1. Check Paley graphs
    paley_primes = [5, 13, 17, 29, 37]
    for p in paley_primes:
        if time.time() > _DEADLINE - 10:
            break
        bits = _paley_graph(p)
        if bits is None:
            continue
        if _is_valid(p, bits):
            edges = _edges_from_bits(p, bits)
            if p > best['n']:
                best = {'n': p, 'edges': edges}
    
    # 2. Try GA for larger n, starting from best known or random
    # We know Paley 17 is valid. Let's try to evolve towards 20, 25, 30, 35.
    target_ns = [20, 25, 30, 35]
    
    for n in target_ns:
        if time.time() > _DEADLINE - 5:
            break
            
        # Try a few GA runs with different seeds
        for attempt in range(3):
            if time.time() > _DEADLINE - 3:
                break
            
            seed_bits = None
            # If we have a smaller valid graph, maybe we can't easily embed it.
            # Just start random or from a sparse/dense bias.
            
            # Allocate time based on remaining budget
            time_left = _DEADLINE - time.time()
            time_alloc = max(1.0, min(5.0, time_left / 3.0))
            
            bits = _ga_search(n, pop_size=30, generations=200, time_limit=time_alloc)
            
            if bits and _is_valid(n, bits):
                edges = _edges_from_bits(n, bits)
                if n > best['n']:
                    best = {'n': n, 'edges': edges}
                    break # Found valid for this n, move to next n
    
    return best

def construct():
    t0 = time.time()
    
    # Start with a known valid baseline (Paley 17)
    best = {'n': 17, 'edges': []}
    bits17 = _paley_graph(17)
    if bits17 is not None and _is_valid(17, bits17):
        best = {'n': 17, 'edges': _edges_from_bits(17, bits17)}
    
    # Try structured search
    try:
        improved = _search_structured()
        if improved['n'] > best['n']:
            best = improved
    except Exception:
        pass
    
    # Safety: ensure validity of what we return
    # Re-verify final result
    if best['n'] > 0:
        # Convert back to bits for fast check
        bits = [0] * best['n']
        for u, v in best['edges']:
            bits[u] |= (1 << v)
            bits[v] |= (1 << u)
        
        if not _is_valid(best['n'], bits):
            # Fallback to Paley 17
            bits17 = _paley_graph(17)
            if bits17 is not None:
                best = {'n': 17, 'edges': _edges_from_bits(17, bits17)}
            else:
                best = {'n': 1, 'edges': []}
    
    return best

if __name__ == "__main__":
    res = construct()
    print(f"Found graph with n={res['n']}")
