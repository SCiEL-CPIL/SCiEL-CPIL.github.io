import time
import random
import math
from itertools import combinations

DEADLINE = 140  # Safety margin for 150s kill

def _compute_score(n, edges):
    """Compute the largest degree sum over triangles using bitmasks."""
    adj_mask = [0] * n
    for u, v in edges:
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
    
    deg = [bin(adj_mask[i]).count('1') for i in range(n)]
    
    max_triangle_sum = 0
    # Iterate over edges; for each, find common neighbors via bitmask AND
    for u, v in edges:
        if u > v:
            u, v = v, u
        common = adj_mask[u] & adj_mask[v]
        # Only consider w > v to avoid double counting
        common &= ~((1 << (v + 1)) - 1)
        while common:
            # Extract lowest set bit
            w_bit = common & (-common)
            w = w_bit.bit_length() - 1
            s = deg[u] + deg[v] + deg[w]
            if s > max_triangle_sum:
                max_triangle_sum = s
            common ^= w_bit
    return max_triangle_sum

def _get_all_pairs(n):
    return [(i, j) for i in range(n) for j in range(i+1, n)]

def _compute_score_with_triangles(n, edges):
    """Compute score and also return list of (triangle, sum) for targeted moves."""
    adj_mask = [0] * n
    for u, v in edges:
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
    
    deg = [bin(adj_mask[i]).count('1') for i in range(n)]
    
    max_triangle_sum = 0
    triangles = []
    for u, v in edges:
        if u > v:
            u, v = v, u
        common = adj_mask[u] & adj_mask[v]
        common &= ~((1 << (v + 1)) - 1)
        while common:
            w_bit = common & (-common)
            w = w_bit.bit_length() - 1
            s = deg[u] + deg[v] + deg[w]
            triangles.append(((u, v, w), s))
            if s > max_triangle_sum:
                max_triangle_sum = s
            common ^= w_bit
    return max_triangle_sum, triangles

def construct():
    start_time = time.time()
    n = 16
    m_target = 65
    
    all_pairs = _get_all_pairs(n)
    
    best_score = float('inf')
    best_edges = None
    
    # Run multiple restarts of SA with different strategies
    max_restarts = 15
    iterations_per_restart = 30000
    
    for restart in range(max_restarts):
        if time.time() - start_time > DEADLINE - 10:
            break
            
        # Initialize graph
        if restart == 0:
            # K_8,8 + 1 internal edge
            edges_set = set()
            for u in range(8):
                for v in range(8, 16):
                    edges_set.add((u, v))
            edges_set.add((0, 1))
        elif restart == 1:
            # K_8,8 + different internal edge
            edges_set = set()
            for u in range(8):
                for v in range(8, 16):
                    edges_set.add((u, v))
            edges_set.add((3, 5))
        elif restart == 2:
            # K_8,8 + 2 internal edges, remove 1 cross edge
            edges_set = set()
            for u in range(8):
                for v in range(8, 16):
                    edges_set.add((u, v))
            edges_set.add((0, 1))
            edges_set.add((2, 3))
            # Remove one cross edge to keep count at 65
            edges_set.discard((0, 8))
        else:
            # Random graph with 65 edges
            random.shuffle(all_pairs)
            edges_set = set(all_pairs[:65])
        
        edges_set = set(edges_set)
        current_score = _compute_score(n, list(edges_set))
        
        if current_score < best_score:
            best_score = current_score
            best_edges = list(edges_set)
        
        # Simulated Annealing with targeted moves
        T_start = 3.0
        T_end = 0.01
        
        for i in range(iterations_per_restart):
            if time.time() - start_time > DEADLINE - 10:
                break
                
            progress = i / iterations_per_restart
            T = T_start * math.pow(T_end / T_start, progress)
            
            # Targeted move: with 30% probability, remove an edge from a high-degree-sum triangle
            # and add a random non-edge
            if random.random() < 0.3 and current_score > 0:
                # Get triangles
                _, triangles = _compute_score_with_triangles(n, list(edges_set))
                if triangles:
                    # Pick a triangle with high sum
                    triangles.sort(key=lambda x: x[1], reverse=True)
                    # Pick from top few
                    top_k = min(5, len(triangles))
                    tri, tri_sum = random.choice(triangles[:top_k])
                    # Pick one of the three edges in this triangle to remove
                    e1_candidates = [(tri[0], tri[1]), (tri[1], tri[2]), (tri[0], tri[2])]
                    e1 = random.choice(e1_candidates)
                    e1 = (min(e1[0], e1[1]), max(e1[0], e1[1]))
                    if e1 not in edges_set:
                        # Fallback to random move
                        e1 = random.choice(list(edges_set))
            else:
                e1 = random.choice(list(edges_set))
            
            # Get a random non-edge to add
            attempts = 0
            e2 = None
            while attempts < 20:
                candidate = random.choice(all_pairs)
                if candidate not in edges_set:
                    e2 = candidate
                    break
                attempts += 1
            
            if e2 is None:
                continue
            
            # Perform move
            new_edges_set = set(edges_set)
            new_edges_set.remove(e1)
            new_edges_set.add(e2)
            
            # Calculate new score
            new_score = _compute_score(n, list(new_edges_set))
            
            # Acceptance
            if new_score <= current_score:
                current_score = new_score
                edges_set = new_edges_set
                if current_score < best_score:
                    best_score = current_score
                    best_edges = list(edges_set)
            else:
                delta = new_score - current_score
                if delta > 0:
                    prob = math.exp(-delta / T)
                    if random.random() < prob:
                        current_score = new_score
                        edges_set = new_edges_set
    
    # Final polish: local search with very low temperature on best found
    if best_edges is not None and time.time() - start_time < DEADLINE - 5:
        edges_set = set(best_edges)
        current_score = best_score
        polish_iters = 20000
        T_start = 0.5
        T_end = 0.01
        
        for i in range(polish_iters):
            if time.time() - start_time > DEADLINE - 3:
                break
                
            progress = i / polish_iters
            T = T_start * math.pow(T_end / T_start, progress)
            
            # Try swapping: remove one edge, add another
            e1 = random.choice(list(edges_set))
            
            attempts = 0
            e2 = None
            while attempts < 20:
                candidate = random.choice(all_pairs)
                if candidate not in edges_set:
                    e2 = candidate
                    break
                attempts += 1
            
            if e2 is None:
                continue
            
            new_edges_set = set(edges_set)
            new_edges_set.remove(e1)
            new_edges_set.add(e2)
            new_score = _compute_score(n, list(new_edges_set))
            
            if new_score <= current_score:
                current_score = new_score
                edges_set = new_edges_set
                if current_score < best_score:
                    best_score = current_score
                    best_edges = list(edges_set)
            else:
                delta = new_score - current_score
                if delta > 0:
                    prob = math.exp(-delta / T)
                    if random.random() < prob:
                        current_score = new_score
                        edges_set = new_edges_set
    
    # Fallback
    if best_edges is None:
        edges_set = set()
        for u in range(8):
            for v in range(8, 16):
                edges_set.add((u, v))
        edges_set.add((0, 1))
        best_edges = list(edges_set)
        
    return {'n': n, 'edges': best_edges}
