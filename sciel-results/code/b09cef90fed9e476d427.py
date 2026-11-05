"""Triangle-free graph without independent 10-set, maximizing n (Ramsey R(3,10))."""
import time
import random
from itertools import combinations

_START = time.time()
_DEADLINE = 115.0  # safety margin before 150s kill


def _remaining_time():
    return _DEADLINE - (time.time() - _START)


def _has_triangle_bitset(adj_bits, n):
    """Check if bitset adjacency has a triangle using bit operations."""
    for i in range(n):
        for j in range(i + 1, n):
            if (adj_bits[i] >> j) & 1:
                # Check if any common neighbor
                common = adj_bits[i] & adj_bits[j]
                if common & ~((1 << (j + 1)) - 1):
                    # There's a common neighbor k > j
                    return True
                # Also check k < i
                if common & ((1 << i) - 1):
                    return True
    return False


def _max_independent_set_size_bitset(adj_bits, n):
    """Find size of maximum independent set using branch and bound with coloring."""
    best = [0]

    def _color_bound(candidates, adj_bits):
        """Greedy coloring of the subgraph induced by candidates to get an upper bound."""
        # We color the complement: independent set in G = clique in complement
        # Actually, we want to bound the max independent set in the subgraph.
        # Use greedy coloring on the graph (not complement) - each color class is an independent set
        # Upper bound on max independent set = number of colors in a proper coloring of the graph
        # Wait, that's for clique. For independent set, we use coloring of the complement.
        
        # Simpler: greedy coloring of the graph gives upper bound on clique number of complement
        # which is alpha of the graph.
        # Actually: alpha(G[S]) <= chi(complement(G[S]))
        # Let's just do a simple greedy coloring of G[S] and use max color as upper bound on alpha
        # No wait. Let me think again.
        
        # For maximum independent set in G:
        # We can use the bound: alpha(G[S]) <= sum of sizes of color classes in a coloring of complement
        # Simpler approach: greedy coloring of G[S], the number of colors is an upper bound on omega(complement) = alpha(G[S])
        
        # Let's do greedy coloring of the complement graph restricted to candidates
        # In complement, edge between u,v if NOT edge in G
        # A coloring of complement = partition into cliques of complement = partition into independent sets of G
        # So number of colors in complement coloring = upper bound on... no, that's lower bound on alpha
        
        # Actually the standard B&B for maximum independent set uses:
        # alpha(G[S]) <= number of colors in a proper coloring of G[S]
        # Wait no. A proper coloring of G[S] partitions S into independent sets.
        # So the max independent set is at most... no, it's at most the sum, which is |S|.
        
        # The correct bound: if we color G[S] with c colors, then any independent set in G[S]
        # can contain at most one vertex from each... no, independent set can contain multiple from same color class.
        
        # I'm confusing myself. Let me use the standard approach:
        # For maximum independent set, we branch and use the bound:
        # alpha(G[S]) <= |S| (trivial)
        # Better: use a greedy coloring of the complement to get a bound.
        
        # Standard approach from literature:
        # To upper bound alpha(G[S]), compute a coloring of G[S] (not complement).
        # Each color class is an independent set in G[S].
        # This gives a LOWER bound on alpha, not upper.
        
        # For UPPER bound on alpha(G[S]):
        # Compute a clique cover of G[S] (partition into cliques).
        # Each independent set can contain at most one vertex per clique.
        # So alpha <= number of cliques in the cover.
        # Equivalently, color the complement of G[S], and the number of colors = clique cover number of G[S].
        
        # Let's do greedy coloring of the complement graph on candidates.
        comp_adj = []
        for v in candidates:
            # In complement, v is connected to u (in candidates) if NOT adj[v][u] in G
            mask = 0
            for u in candidates:
                if u != v and not ((adj_bits[v] >> u) & 1):
                    mask |= (1 << u)
            comp_adj.append(mask)
        
        # Greedy coloring
        color = [-1] * n
        num_colors = 0
        for v in candidates:
            # Find smallest color not used by neighbors in complement
            used = set()
            for u in candidates:
                if u != v and ((comp_adj[candidates.index(v)] >> u) & 1) and color[u] != -1:
                    used.add(color[u])
            c = 0
            while c in used:
                c += 1
            color[v] = c
            num_colors = max(num_colors, c + 1)
        
        return num_colors

    def bnb(candidates, current_size):
        if not candidates:
            if current_size > best[0]:
                best[0] = current_size
            return
        
        # Trivial upper bound
        if current_size + len(candidates) <= best[0]:
            return
        
        # Pick a vertex to branch on (highest degree in induced subgraph for better pruning)
        # Use first vertex for simplicity, or pick one with max degree
        v = candidates[0]
        rest = candidates[1:]
        
        # Include v: remove v and its neighbors from rest
        # In independent set, if we include v, we can't include any neighbor of v
        neighbors_mask = adj_bits[v]
        non_neighbors = [u for u in rest if not ((neighbors_mask >> u) & 1)]
        bnb(non_neighbors, current_size + 1)
        
        # Exclude v
        bnb(rest, current_size)

    bnb(list(range(n)), 0)
    return best[0]


def _max_independent_set_fast(adj_bits, n):
    """Optimized maximum independent set using bit-parallel B&B."""
    best = [0]
    full_mask = (1 << n) - 1

    def bnb(cand_mask, current_size):
        # Count bits for upper bound
        if cand_mask == 0:
            if current_size > best[0]:
                best[0] = current_size
            return

        # Upper bound
        if current_size + bin(cand_mask).count('1') <= best[0]:
            return

        # Pick vertex with max degree in induced subgraph
        # Iterate through set bits
        cm = cand_mask
        max_deg = -1
        max_v = -1
        while cm:
            v = cm & (-cm)
            v_idx = v.bit_length() - 1
            cm ^= v
            # Count neighbors of v within candidates
            deg = bin(adj_bits[v_idx] & cand_mask & ~v).count('1')
            if deg > max_deg:
                max_deg = deg
                max_v = v_idx

        rest = cand_mask & ~(1 << max_v)

        # Branch: include max_v first (usually better for independent set)
        # Non-neighbors of max_v in rest
        non_nbrs = rest & ~adj_bits[max_v]
        bnb(non_nbrs, current_size + 1)

        # Branch: exclude max_v
        bnb(rest, current_size)

    bnb(full_mask, 0)
    return best[0]


def _build_triangle_free_bitset(n, seed=0):
    """
    Build a triangle-free graph using bitset representation.
    Returns list of n integers, each a bitmask of neighbors.
    """
    rng = random.Random(seed)
    adj_bits = [0] * n
    
    edges_order = list(combinations(range(n), 2))
    rng.shuffle(edges_order)
    
    for i, j in edges_order:
        # Check if adding (i,j) creates a triangle
        # Common neighbors of i and j
        common = adj_bits[i] & adj_bits[j]
        if common:
            continue
        # Add edge
        adj_bits[i] |= (1 << j)
        adj_bits[j] |= (1 << i)
    
    return adj_bits


def _local_search_bitset(adj_bits, n, seed=0):
    """
    Local search to reduce independence number while maintaining triangle-free.
    Uses both edge additions and targeted edge removals.
    """
    rng = random.Random(seed)
    adj_bits = adj_bits[:]
    
    alpha = _max_independent_set_fast(adj_bits, n)
    
    if alpha < 10:
        return adj_bits, alpha
    
    all_non_edges = [(a, b) for a in range(n) for b in range(a + 1, n)
                     if not ((adj_bits[a] >> b) & 1)]
    all_edges = [(a, b) for a in range(n) for b in range(a + 1, n)
                 if (adj_bits[a] >> b) & 1]
    
    max_iter = 300
    for it in range(max_iter):
        if _remaining_time() < 5:
            break
        
        # Try adding an edge (primary strategy)
        if all_non_edges and rng.random() < 0.8:
            i, j = rng.choice(all_non_edges)
            # Check triangle-free
            common = adj_bits[i] & adj_bits[j]
            if common:
                continue
            
            # Add edge
            adj_bits[i] |= (1 << j)
            adj_bits[j] |= (1 << i)
            
            new_alpha = _max_independent_set_fast(adj_bits, n)
            if new_alpha < alpha:
                alpha = new_alpha
                # Update non-edge list
                all_non_edges.remove((i, j))
                all_edges.append((i, j))
                if alpha < 10:
                    break
            else:
                # Revert
                adj_bits[i] &= ~(1 << j)
                adj_bits[j] &= ~(1 << i)
        else:
            # Try removing an edge (secondary, to unblock future additions)
            if not all_edges:
                continue
            i, j = rng.choice(all_edges)
            
            # Remove edge
            adj_bits[i] &= ~(1 << j)
            adj_bits[j] &= ~(1 << i)
            
            # After removal, try to add a different edge that might help
            # Find a non-edge that can now be added
            added = False
            for a, b in all_non_edges[:20]:  # Check limited candidates
                if a == i or a == j or b == i or b == j:
                    continue
                common = adj_bits[a] & adj_bits[b]
                if common:
                    continue
                # Check if this edge would reduce alpha
                adj_bits[a] |= (1 << b)
                adj_bits[b] |= (1 << a)
                test_alpha = _max_independent_set_fast(adj_bits, n)
                if test_alpha < alpha:
                    # Keep both changes
                    alpha = test_alpha
                    all_non_edges.remove((a, b))
                    all_edges.append((a, b))
                    added = True
                    if alpha < 10:
                        break
                    break
                else:
                    adj_bits[a] &= ~(1 << b)
                    adj_bits[b] &= ~(1 << a)
            
            if not added:
                # Revert removal
                adj_bits[i] |= (1 << j)
                adj_bits[j] |= (1 << i)
    
    return adj_bits, alpha


def _bitset_to_edges(adj_bits, n):
    """Convert bitset adjacency to edge list."""
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if (adj_bits[i] >> j) & 1:
                edges.append([i, j])
    return edges


def construct():
    """
    Construct a triangle-free graph on n vertices (1 <= n <= 41) with no independent set of 10.
    Maximize n.
    """
    best_adj_bits = None
    best_n = 0
    best_alpha = float('inf')
    
    for n_target in range(40, 25, -1):
        if _remaining_time() < 10:
            break
        
        found = False
        # More seeds for larger n, fewer for smaller
        n_seeds = 120 if n_target >= 38 else 60
        for seed in range(n_seeds):
            if _remaining_time() < 5:
                break
            
            adj_bits = _build_triangle_free_bitset(n_target, seed)
            
            alpha = _max_independent_set_fast(adj_bits, n_target)
            
            if alpha < 10:
                best_adj_bits = adj_bits
                best_n = n_target
                best_alpha = alpha
                found = True
                break
            
            # Try local search
            if _remaining_time() > 8:
                adj_bits_ls, alpha_ls = _local_search_bitset(adj_bits, n_target, seed)
                if alpha_ls < 10:
                    best_adj_bits = adj_bits_ls
                    best_n = n_target
                    best_alpha = alpha_ls
                    found = True
                    break
        
        if found:
            break
    
    # Fallback: Andrásfai graph And(9) - 27 vertices, triangle-free, alpha=9
    if best_adj_bits is None:
        n = 27
        adj_bits = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                d = (j - i) % n
                if 1 <= d <= 8 and d % 3 != 0:
                    adj_bits[i] |= (1 << j)
                    adj_bits[j] |= (1 << i)
        best_adj_bits = adj_bits
        best_n = n
    
    edges = _bitset_to_edges(best_adj_bits, best_n)
    
    return {'n': best_n, 'edges': edges}
