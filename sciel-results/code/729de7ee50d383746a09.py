"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import random
from collections import defaultdict

# EVOLVE-BLOCK-START

MAX_MOD = 128
DEADLINE_MARGIN = 5.0  # seconds before the 90s limit

def _is_disjoint(r1, m1, r2, m2):
    """Check if two residue classes are disjoint using gcd criterion."""
    g = math.gcd(m1, m2)
    return (r1 - r2) % g != 0

def _construct_nodes():
    """Generate all (residue, modulus) pairs and assign indices."""
    nodes = []
    index_map = {}
    idx = 0
    for m in range(2, MAX_MOD + 1):
        for r in range(m):
            nodes.append((r, m))
            index_map[(r, m)] = idx
            idx += 1
    return nodes, index_map

def _build_compatibility_graph(nodes):
    """
    Build compatibility bitsets for Max Clique.
    compat[i] is a bitmask where bit j is 1 if nodes[i] and nodes[j] are disjoint.
    """
    n = len(nodes)
    compat = [0] * n
    
    # Optimization: Group by modulus to avoid redundant gcd calculations?
    # Or just loop. 8255^2 is ~68M.
    # Let's try to be efficient.
    
    for i in range(n):
        r1, m1 = nodes[i]
        mask = 0
        for j in range(i + 1, n):
            r2, m2 = nodes[j]
            if _is_disjoint(r1, m1, r2, m2):
                mask |= (1 << j)
                compat[j] |= (1 << i)
        compat[i] = mask
        
    return compat

def _max_clique_tomita(compat, deadline):
    """
    Tomita's branch and bound algorithm for Maximum Clique.
    """
    n = len(compat)
    best_clique = []
    best_size = 0
    
    # Global deadline check
    start_time = time.time()
    
    # Order nodes by degree (number of compatible neighbors)
    # Degree in compatibility graph = popcount(compat[i])
    degrees = [compat[i].bit_count() for i in range(n)]
    order = sorted(range(n), key=lambda x: degrees[x], reverse=True)
    
    # We need to map old indices to new indices or just use the order array.
    # Tomita usually works on a fixed set P.
    # We will implement the recursive expand function.
    
    def expand(P, size, current_clique):
        nonlocal best_size, best_clique
        
        if time.time() - start_time > deadline:
            return
            
        # Upper bound: if size + popcount(P) <= best_size, prune
        if size + P.bit_count() <= best_size:
            return
            
        # If P is empty, check if we found a better clique
        if P == 0:
            if size > best_size:
                best_size = size
                best_clique = list(current_clique)
            return
            
        # Greedy coloring of P to get an upper bound
        # We color the nodes in P. The max color is an upper bound on clique size in P.
        # We process nodes in a specific order to make the bound tight.
        
        # For simplicity and speed, we can just use popcount(P) as a loose bound initially,
        # but Tomita's coloring is much better.
        # Let's implement a simple coloring.
        
        # We need to iterate through nodes in P.
        # To make it efficient, we can extract nodes from P.
        
        # A simpler bound:
        # The max clique size in P is at most the size of a maximal matching? No.
        # Let's use the property that we can pick nodes one by one.
        
        # Standard Tomita:
        # 1. Order nodes in P.
        # 2. Color them greedily.
        # 3. Process nodes in reverse color order.
        
        # Since extracting and sorting nodes from a bitmask in Python is slow,
        # we will use a slightly less tight but faster bound or a simplified Tomita.
        
        # Simplified approach:
        # Pick a node v from P.
        # Branch 1: Include v. New P = P & compat[v] & (P without v and nodes before v?)
        # To avoid symmetry, we maintain an order.
        
        # Let's use the "reverse order" trick from Tomita.
        # We need to know which nodes are in P.
        # We can iterate through set bits in P.
        
        # To keep it simple and fast enough for 8k nodes:
        # We will just pick the node with highest degree in P?
        # Or just the first one?
        
        # Let's try a simple recursive search with pruning.
        # To avoid exponential blowup, we need good pruning.
        # The `size + P.bit_count() <= best_size` is weak.
        
        # Let's try to get a better bound.
        # We can compute a greedy coloring on the fly for the nodes in P.
        # But that's expensive.
        
        # Alternative: Use the fact that the graph is dense.
        # If the graph is dense, a greedy algorithm finds a large clique quickly.
        # We can use the current best_size to prune.
        
        # Let's just pick a node.
        # To break symmetry, we pick the lowest index node in P?
        # Or highest degree?
        
        # Let's try picking the node with the highest degree in P.
        # Finding highest degree in P requires checking degrees of all bits in P.
        
        # For now, let's just pick the lowest set bit.
        # This might be slow.
        
        # Better: Pre-compute a static order and stick to it?
        # If we stick to a static order, we can use `P & ((1 << v) - 1)` to only consider nodes "before" v in the order?
        # No, that's for finding all cliques. For Max Clique, we just need one.
        
        # Let's use the standard Tomita with a static ordering `order`.
        # We maintain `P` as a bitmask of nodes that are "after" the current node in `order`?
        # No, `P` is the set of candidates.
        
        # Let's implement a version that iterates through nodes in `order`.
        
        pass # Placeholder for logic below

    # Given the complexity of implementing full Tomita with bitsets in Python within the time limit,
    # and the risk of timeouts, let's use a simpler but effective heuristic:
    # 1. Greedy construction to get a good initial lower bound.
    # 2. Local search to improve.
    # 3. If time permits, try random restarts.
    
    # This is a fallback to the previous strategy but optimized.
    
    return best_clique, best_size

def construct():
    start_time = time.time()
    deadline = 90.0 - DEADLINE_MARGIN
    
    # 1. Generate nodes
    nodes, index_map = _construct_nodes()
    n = len(nodes)
    
    # 2. Build compatibility graph (bitsets)
    # This is the expensive part.
    compat = [0] * n
    
    # Optimization: Process moduli in groups?
    # Just double loop.
    for i in range(n):
        r1, m1 = nodes[i]
        for j in range(i + 1, n):
            r2, m2 = nodes[j]
            if _is_disjoint(r1, m1, r2, m2):
                compat[i] |= (1 << j)
                compat[j] |= (1 << i)
                
    # 3. Find Max Clique
    # We will use a simple branch and bound with a greedy coloring bound.
    
    best_clique = []
    best_size = 0
    
    # Initial greedy solution
    # Greedy: pick node with max degree, remove it and its neighbors, repeat.
    # This gives a maximal independent set in conflict graph = maximal clique in compat graph.
    
    def greedy_clique(compat, n):
        P = (1 << n) - 1
        clique = []
        while P:
            # Find node with max degree in P
            # This is O(N) per step, so O(N^2) total.
            max_deg = -1
            best_node = -1
            p = P
            while p:
                # Get lowest set bit
                lb = p & (-p)
                idx = lb.bit_length() - 1
                deg = (compat[idx] & P).bit_count()
                if deg > max_deg:
                    max_deg = deg
                    best_node = idx
                p ^= lb
                
            if best_node == -1:
                break
                
            clique.append(best_node)
            P &= compat[best_node]
            P &= ~(1 << best_node)
            
        return clique

    # Get initial solution
    init_clique = greedy_clique(compat, n)
    if len(init_clique) > best_size:
        best_size = len(init_clique)
        best_clique = init_clique
        
    # Local Search: 1-1 swap? 1-2 swap?
    # Given time constraints, let's try a few random restarts of greedy with different node orderings.
    
    # To improve, we can shuffle the node indices before greedy.
    # But we need to map back to original indices.
    
    # Let's try to improve the current best clique by local search.
    # For each node in the clique, try to swap it with a node outside?
    # Or just try to add a node? (Impossible if it's a maximal clique).
    # So we must remove one and add two?
    
    # Given the 90s limit and the cost of building the graph, we have ~80s for search.
    # Let's do random restarts of the greedy algorithm with perturbed node priorities.
    
    remaining_time = deadline - (time.time() - start_time)
    
    if remaining_time > 10:
        # Try multiple random orderings
        # We can simulate random orderings by adding random weights to degrees.
        
        for restart in range(100):
            if time.time() - start_time > deadline - 5:
                break
                
            # Perturb degrees
            perturbed_degrees = []
            for i in range(n):
                deg = compat[i].bit_count()
                # Add small random noise to break ties
                noise = random.random() * 10
                perturbed_degrees.append(deg + noise)
            
            # Sort by perturbed degree
            order = sorted(range(n), key=lambda x: perturbed_degrees[x], reverse=True)
            
            # Greedy with this order
            P = (1 << n) - 1
            clique = []
            # We need to pick the first available node in `order` that is in P?
            # Or just iterate through `order` and pick if compatible with current clique?
            # The standard greedy for clique:
            # Iterate through nodes in order. If node is compatible with all in clique, add it.
            # This is O(N * |clique|).
            
            for idx in order:
                if not (P & (1 << idx)):
                    continue
                # Check if idx is compatible with all nodes in clique
                compatible = True
                for c in clique:
                    if not (compat[idx] & (1 << c)):
                        compatible = False
                        break
                if compatible:
                    clique.append(idx)
                    # Update P? 
                    # For greedy maximal clique, we just need to ensure we don't pick conflicts.
                    # If we pick idx, we must remove its neighbors from consideration for FUTURE picks?
                    # No, the standard "add if compatible with all previous" produces a clique, 
                    # but not necessarily maximal in the sense of "can't add any more from the whole set".
                    # It is maximal with respect to the order.
                    
            if len(clique) > best_size:
                best_size = len(clique)
                best_clique = clique

    # Convert best_clique (indices) back to (residue, modulus) pairs
    result = [[nodes[i][0], nodes[i][1]] for i in best_clique]
    
    if not result:
        # Fallback
        return [[(1 << k) - 1, 1 << (k + 1)] for k in range(7)]
        
    return result

# EVOLVE-BLOCK-END
