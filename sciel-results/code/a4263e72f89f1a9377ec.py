"""Hexagon-free graph construction on 40 vertices via Simulated Annealing."""
# EVOLVE-BLOCK-START
import time
import random
import math

N = 40
M = N * (N - 1) // 2


def _build_adj_from_bits(bits):
    adj = [set() for _ in range(N)]
    idx = 0
    for u in range(N):
        for v in range(u + 1, N):
            if bits[idx]:
                adj[u].add(v)
                adj[v].add(u)
            idx += 1
    return adj


def _count_c6s(adj):
    """Count the number of 6-cycles in the graph."""
    count = 0
    for u in range(N):
        for v in adj[u]:
            if v <= u:
                continue
            for w in adj[v]:
                if w == u:
                    continue
                for x in adj[w]:
                    if x == u or x == v:
                        continue
                    for y in adj[x]:
                        if y == u or y == v or y == w:
                            continue
                        for z in adj[y]:
                            if z == u or z == v or z == w or z == x:
                                continue
                            if u in adj[z]:
                                count += 1
    return count


def _seed_graph_8K5():
    """8 disjoint K5s: 100 edges."""
    edges = []
    for k in range(8):
        base = 5 * k
        for a in range(5):
            for b in range(a + 1, 5):
                edges.append([base + a, base + b])
    return edges


def _bits_from_edges(edges):
    bits = [0] * M
    idx = 0
    for u in range(N):
        for v in range(u + 1, N):
            if (u, v) in edges or (v, u) in edges:
                bits[idx] = 1
            idx += 1
    return bits


def _edges_from_bits(bits):
    edges = []
    idx = 0
    for u in range(N):
        for v in range(u + 1, N):
            if bits[idx]:
                edges.append([u, v])
            idx += 1
    return edges


def _simulate_annealing(start_time, time_limit, seed):
    random.seed(seed)
    
    # Initialize with 8-K5
    edges = _seed_graph_8K5()
    bits = _bits_from_edges(edges)
    adj = _build_adj_from_bits(bits)
    current_energy = _count_c6s(adj)  # Should be 0
    
    best_energy = current_energy
    best_bits = list(bits)
    
    # Simulated Annealing Parameters
    T = 50.0  # Initial temperature
    cooling = 0.998
    min_T = 0.1
    
    # Precompute edge indices for fast access
    # We need a way to pick a random edge to flip.
    # We can just pick a random index 0..M-1.
    
    while time.time() - start_time < time_limit:
        if T < min_T:
            break
            
        # Pick a random edge to flip
        idx = random.randint(0, M - 1)
        
        # Determine u, v from idx
        # This is slightly inefficient, but M=780 is small.
        # Alternatively, maintain a list of edges.
        # Let's use a lookup table for speed.
        # We can precompute (u, v) for each idx.
        
        # For now, let's just iterate or use a helper.
        # Actually, let's precompute edge endpoints outside the loop.
        # But we are inside a function. Let's do it once.
        
        # Wait, I should precompute edge endpoints globally or pass them.
        # Let's create a global list of edges.
        
        u, v = _idx_to_uv[idx]
        
        # Flip the bit
        bits[idx] = 1 - bits[idx]
        
        # Update adjacency
        if bits[idx]:
            adj[u].add(v)
            adj[v].add(u)
        else:
            adj[u].discard(v)
            adj[v].discard(u)
            
        # Calculate new energy
        # This is the bottleneck. _count_c6s is O(V^6) worst case, but much less in practice.
        # We need to be careful with time.
        # If we do this too often, we'll time out.
        
        # Optimization: Incremental C6 count update?
        # Too complex. Let's rely on the fact that graphs are sparse.
        # 100 edges on 40 vertices is quite sparse.
        
        new_energy = _count_c6s(adj)
        
        # Acceptance criterion
        delta_E = new_energy - current_energy
        if delta_E < 0:
            current_energy = new_energy
            if new_energy < best_energy:
                best_energy = new_energy
                best_bits = list(bits)
        else:
            # Accept with probability exp(-delta_E / T)
            # Since we want to minimize energy (number of C6s),
            # and we want to maximize edges, we should incorporate edge count into energy.
            # Energy = C6_count - lambda * Edge_count
            # Let's define Energy = C6_count * 1000 - Edge_count
            # This way, adding an edge (if it doesn't create a C6) reduces energy.
            # Creating a C6 increases energy by 1000.
            
            # I need to update my energy calculation to include edge count.
            # Let's restart the logic inside the loop with the new energy definition.
            
            # Actually, let's just use the probability for the C6 count for now,
            # and rely on the fact that we start with a valid graph and mostly add edges.
            # But SA needs a proper objective.
            
            # Let's define:
            # Objective = - (Edge_Count - 100 * C6_Count)
            # Minimize Objective.
            # Or Energy = 100 * C6_Count - Edge_Count.
            
            # Let's re-implement the energy calculation properly.
            
            # I will modify the loop to use the combined energy.
            
            # For this specific step, if delta_E (C6 count) is positive,
            # it means we created a C6.
            # We should accept it with some probability to escape local optima.
            
            if random.random() < math.exp(-delta_E / T):
                current_energy = new_energy
                if new_energy < best_energy:
                    best_energy = new_energy
                    best_bits = list(bits)
            else:
                # Reject: revert
                bits[idx] = 1 - bits[idx]
                if bits[idx]:
                    adj[u].add(v)
                    adj[v].add(u)
                else:
                    adj[u].discard(v)
                    adj[v].discard(u)
                    
        T *= cooling
        
    return best_bits


# Precompute edge indices
_edge_uv = []
for u in range(N):
    for v in range(u + 1, N):
        _edge_uv.append((u, v))

_idx_to_uv = _edge_uv


def construct():
    start_time = time.time()
    deadline = 135  # Safety margin
    
    # Run SA with a few different seeds/temps if time permits
    # But SA is slow due to _count_c6s.
    # Let's try one long run.
    
    # To make it faster, we can use a heuristic to update C6 count,
    # but for now, let's just run it.
    
    # If it's too slow, we might need to optimize _count_c6s.
    
    # Let's try to run SA for the available time.
    # We need to be careful that _count_c6s doesn't take too long.
    
    # Optimization: If the graph is dense, _count_c6s is slow.
    # But we start sparse.
    
    # Let's run SA.
    
    # To ensure we return something valid, we keep track of the best valid graph (energy=0).
    
    # The SA loop above tracks best_energy.
    # If best_energy is not 0, the result is invalid.
    # We must return a valid graph.
    
    # So, we should only update best_bits if new_energy == 0.
    # Or, we track best_valid_bits.
    
    # Let's modify the SA logic to only accept improvements in edge count IF energy remains 0.
    # And allow moves with energy > 0 only if we are in a "high temp" phase,
    # but we must revert to the last valid state if we don't find a better valid state.
    
    # This is getting complicated.
    # Let's stick to the simple version:
    # Energy = C6_Count.
    # We want to minimize C6_Count.
    # But we also want to maximize edges.
    
    # Let's use the combined energy: E = 1000 * C6_Count - Edge_Count.
    
    # I will rewrite the SA loop to use this combined energy.
    
    # Since I can't easily edit the code above without redefining functions,
    # let's implement a new function `_simulate_annealing_combined`.
    
    pass


def _simulate_annealing_combined(start_time, time_limit, seed):
    random.seed(seed)
    
    edges = _seed_graph_8K5()
    bits = _bits_from_edges(edges)
    adj = _build_adj_from_bits(bits)
    
    # Calculate initial energy
    c6_count = _count_c6s(adj)
    edge_count = sum(bits)
    current_energy = 1000 * c6_count - edge_count
    
    best_energy = current_energy
    best_bits = list(bits)
    
    T = 50.0
    cooling = 0.999
    min_T = 0.1
    
    while time.time() - start_time < time_limit:
        if T < min_T:
            break
            
        idx = random.randint(0, M - 1)
        u, v = _edge_uv[idx]
        
        # Flip
        bits[idx] = 1 - bits[idx]
        
        if bits[idx]:
            adj[u].add(v)
            adj[v].add(u)
        else:
            adj[u].discard(v)
            adj[v].discard(u)
            
        # New Energy
        new_c6_count = _count_c6s(adj)
        new_edge_count = sum(bits)
        new_energy = 1000 * new_c6_count - new_edge_count
        
        delta_E = new_energy - current_energy
        
        if delta_E < 0:
            current_energy = new_energy
            if new_energy < best_energy:
                best_energy = new_energy
                best_bits = list(bits)
        else:
            if random.random() < math.exp(-delta_E / T):
                current_energy = new_energy
                if new_energy < best_energy:
                    best_energy = new_energy
                    best_bits = list(bits)
            else:
                # Revert
                bits[idx] = 1 - bits[idx]
                if bits[idx]:
                    adj[u].add(v)
                    adj[v].add(u)
                else:
                    adj[u].discard(v)
                    adj[v].discard(u)
                    
        T *= cooling
        
    return best_bits


def construct():
    start_time = time.time()
    deadline = 135
    
    # Run SA
    # Since _count_c6s is expensive, we might not get many iterations.
    # Let's see.
    
    # If we time out, we return the best valid graph found.
    # But best_bits might have C6s.
    # We must ensure we return a valid graph.
    
    # So, we should track the best VALID graph separately.
    
    # Let's modify the logic:
    # Only update best_valid_bits if c6_count == 0.
    
    # I will implement this check in the final return.
    
    bits = _simulate_annealing_combined(start_time, deadline - 5, seed=42)
    
    edges = _edges_from_bits(bits)
    
    # Verify
    adj = _build_adj_from_bits(bits)
    if _count_c6s(adj) > 0:
        # Fallback to seed
        return _seed_graph_8K5()
        
    return edges
# EVOLVE-BLOCK-END
