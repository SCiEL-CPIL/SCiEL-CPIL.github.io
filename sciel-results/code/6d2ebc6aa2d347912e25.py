"""
Linear triple systems on 16 points with small independence number.

We construct a linear 3-uniform hypergraph on 16 vertices (any two edges
share at most one vertex) and minimize the independence number (largest
subset of vertices containing no edge).

Strategy:
- Use a known good base structure derived from projective geometry /
  combinatorial designs, then improve via local search (edge swaps).
- The base: take the 5 disjoint triples from the seed as a starting point,
  then greedily add triples to reduce independence number.
- Actually, a better approach: use a structured construction.

Key insight: The minimum independence number for linear 3-graphs on 16
vertices. We want to find a linear 3-uniform hypergraph on 16 vertices
with the smallest possible independence number.

A lower bound: if we have n=16 vertices and b edges, the independence
number alpha satisfies alpha >= n - (number of vertices covered by edges
in a maximal independent set)... Actually, let's think differently.

For a 3-uniform linear hypergraph, each edge covers 3 vertices. If we
have b edges, the maximum number of vertices that can be "hit" by
selecting one vertex per edge is at most b (but with overlaps). The
independence number is the size of the largest set with no edge fully
contained in it.

Equivalently, alpha = min over all vertex subsets S of |S| such that no
edge is fully in S... no, alpha is the MAXIMUM size of S with no edge
fully in S.

To minimize alpha, we want every large subset to contain at least one edge.

A systematic approach:
1. Start with a structured base (e.g., derived from a 2-(16,4,2) design
   or similar, then take triples from it).
2. Use local search: try swapping edges to reduce independence number.

Actually, let me think about known constructions. For n=16, a Steiner
triple system doesn't exist (16 ≡ 1 mod 6 is required, 16 ≡ 4 mod 6).

Let me use a greedy + local search approach:
- Start with a random linear 3-graph
- Compute independence number via brute force (2^16 = 65536 subsets, feasible)
- Try local moves: add an edge, remove an edge, swap an edge

Since 2^16 = 65536, we can compute the independence number exactly by
iterating over all subsets.

Let me design a more structured approach:

Base construction: Use the affine geometry AG(4,2) which has 16 points.
Lines in AG(4,2) have 2 points (not useful for triples). 
Planes in AG(4,2) have 4 points.

Alternative: Use a 2-(16,4,2) design (a biplane or similar) and derive
triples. Actually, let's just do a smart greedy + local search.

Plan:
1. Initialize with a good seed: use the 5 disjoint triples plus additional
   triples added greedily to minimize independence number.
2. Local search: try all possible single-edge swaps and additions.
3. Compute independence number exactly via bitmask enumeration.
"""

import time
import random
import itertools

def _independence_number(edges):
    """Compute the independence number of a 3-uniform hypergraph on 16 vertices."""
    n = 16
    # Precompute for each vertex, which edges contain it
    # For each subset (as bitmask), check if any edge is fully contained
    # An edge (a,b,c) is contained in mask if (mask & (1<<a)) and (mask & (1<<b)) and (mask & (1<<c))
    
    # Build edge masks
    edge_masks = [0] * len(edges)
    for i, (a, b, c) in enumerate(edges):
        edge_masks[i] = (1 << a) | (1 << b) | (1 << c)
    
    # For each subset size from n down to 0, check if there's an independent set of that size
    # Independent set = no edge fully contained
    # We want the largest k such that there exists a subset of size k with no edge fully in it
    
    # Iterate over all 2^16 = 65536 masks
    # For each mask, check if it's independent (no edge fully contained)
    # Track the max size of an independent set
    
    best = 0
    # Optimization: iterate masks by popcount descending
    # But 65536 is small, just iterate all
    for mask in range(1 << n):
        # Check if mask is independent
        independent = True
        for em in edge_masks:
            if (mask & em) == em:  # edge fully contained in mask
                independent = False
                break
        if independent:
            pc = bin(mask).count('1')
            if pc > best:
                best = pc
                if best == n:
                    break
    return best

def _is_linear(edges):
    """Check if any two edges share at most one vertex."""
    for i in range(len(edges)):
        for j in range(i + 1, len(edges)):
            common = set(edges[i]) & set(edges[j])
            if len(common) > 1:
                return False
    return True

def _can_add_edge(edges, a, b, c):
    """Check if adding edge (a,b,c) maintains linearity."""
    new_edge = {a, b, c}
    for e in edges:
        common = new_edge & set(e)
        if len(common) > 1:
            return False
    return True

def _greedy_build(seed_edges, rng):
    """Greedily add edges to minimize independence number."""
    edges = list(seed_edges)
    
    # Try adding all possible triples that maintain linearity
    # Score each by how much it reduces independence number
    # But computing independence number for each candidate is expensive
    # Instead, use a heuristic: prefer edges that cover "sparse" vertices
    
    # First, let's just add all possible linear edges in a smart order
    all_triples = list(itertools.combinations(range(16), 3))
    rng.shuffle(all_triples)
    
    for (a, b, c) in all_triples:
        if _can_add_edge(edges, a, b, c):
            edges.append([a, b, c])
    
    return edges

def _local_search(edges, rng, deadline):
    """Local search: try edge swaps to reduce independence number."""
    current_alpha = _independence_number(edges)
    best_edges = list(edges)
    best_alpha = current_alpha
    
    all_triples = list(itertools.combinations(range(16), 3))
    
    while time.time() < deadline:
        # Try removing a random edge and adding a different one
        improved = False
        
        # Try: remove one edge, add another
        for i in range(len(best_edges)):
            if time.time() >= deadline:
                break
            
            candidate = best_edges[:i] + best_edges[i+1:]
            
            # Try adding a random edge
            for _ in range(20):
                a, b, c = rng.choice(all_triples)
                if _can_add_edge(candidate, a, b, c):
                    new_edges = candidate + [[a, b, c]]
                    new_alpha = _independence_number(new_edges)
                    if new_alpha < best_alpha:
                        best_alpha = new_alpha
                        best_edges = new_edges
                        improved = True
                        break
        
        # Try: swap two edges
        if not improved:
            for _ in range(50):
                if time.time() >= deadline:
                    break
                i, j = rng.sample(range(len(best_edges)), 2)
                a, b, c = rng.choice(all_triples)
                candidate = best_edges[:i] + best_edges[i+1:]
                # Remove edge j (adjust index)
                if j > i:
                    j -= 1
                candidate = candidate[:j] + candidate[j+1:]
                
                # Try adding new edge
                if _can_add_edge(candidate, a, b, c):
                    new_edges = candidate + [[a, b, c]]
                    new_alpha = _independence_number(new_edges)
                    if new_alpha < best_alpha:
                        best_alpha = new_alpha
                        best_edges = new_edges
                        improved = True
                        break
        
        if not improved:
            # Try adding an edge
            for _ in range(30):
                if time.time() >= deadline:
                    break
                a, b, c = rng.choice(all_triples)
                if _can_add_edge(best_edges, a, b, c):
                    new_edges = best_edges + [[a, b, c]]
                    new_alpha = _independence_number(new_edges)
                    if new_alpha < best_alpha:
                        best_alpha = new_alpha
                        best_edges = new_edges
                        improved = True
                        break
            
            if not improved:
                # Stagnation: try a random restart with perturbation
                if rng.random() < 0.1:
                    # Perturb: remove 2-3 random edges, re-add
                    for _ in range(rng.randint(2, 3)):
                        idx = rng.randrange(len(best_edges))
                        best_edges.pop(idx)
                    # Re-add greedily
                    for (a, b, c) in all_triples:
                        if _can_add_edge(best_edges, a, b, c):
                            best_edges.append([a, b, c])
                    best_alpha = _independence_number(best_edges)
    
    return best_edges, best_alpha

def construct():
    deadline = time.time() + 140  # 10s safety margin
    
    # Start with a structured base
    # Use 5 disjoint triples as base, then greedily fill
    seed_edges = [[3 * i, 3 * i + 1, 3 * i + 2] for i in range(5)]
    
    rng = random.Random(42)
    
    # Greedy fill
    edges = _greedy_build(seed_edges, rng)
    
    # Local search
    edges, alpha = _local_search(edges, rng, deadline)
    
    # Verify linearity
    if not _is_linear(edges):
        # Fallback to seed
        return seed_edges
    
    return edges

# Test
if __name__ == "__main__":
    result = construct()
    print(f"Number of edges: {len(result)}")
    print(f"Independence number: {_independence_number(result)}")
    print(f"Linear: {_is_linear(result)}")
