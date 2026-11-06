import time
import random
import math
from itertools import combinations
import numpy as np

DEADLINE = 140  # Safety margin for 150s kill

def _compute_score(n, edges):
    """Compute the largest degree sum over triangles using numpy bitmasks."""
    adj_mask = [0] * n
    for u, v in edges:
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
    
    deg = [bin(adj_mask[i]).count('1') for i in range(n)]
    
    max_triangle_sum = 0
    for u, v in edges:
        if u > v:
            u, v = v, u
        common = adj_mask[u] & adj_mask[v]
        common &= ~((1 << (v + 1)) - 1)
        while common:
            w_bit = common & (-common)
            w = w_bit.bit_length() - 1
            s = deg[u] + deg[v] + deg[w]
            if s > max_triangle_sum:
                max_triangle_sum = s
            common ^= w_bit
    return max_triangle_sum

def _compute_score_fast(n, adj_mask, deg, edges_list):
    """Faster score computation given precomputed adj_mask and deg."""
    max_triangle_sum = 0
    for u, v in edges_list:
        if u > v:
            u, v = v, u
        common = adj_mask[u] & adj_mask[v]
        common &= ~((1 << (v + 1)) - 1)
        while common:
            w_bit = common & (-common)
            w = w_bit.bit_length() - 1
            s = deg[u] + deg[v] + deg[w]
            if s > max_triangle_sum:
                max_triangle_sum = s
            common ^= w_bit
    return max_triangle_sum

def _get_all_pairs(n):
    return [(i, j) for i in range(n) for j in range(i+1, n)]

def _compute_triangles_info(n, edges):
    """Compute score and return list of (triangle, sum) for targeted moves."""
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
    return max_triangle_sum, triangles, adj_mask, deg

def construct():
    start_time = time.time()
    n = 16
    m_target = 65
    
    all_pairs = _get_all_pairs(n)
    all_pairs_set = set(all_pairs)
    
    best_score = float('inf')
    best_edges = None
    
    max_restarts = 20
    iterations_per_restart = 25000
    
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
            edges_set.discard((0, 8))
        elif restart == 3:
            # K_8,8 + 2 internal edges, remove 1 cross edge (different)
            edges_set = set()
            for u in range(8):
                for v in range(8, 16):
                    edges_set.add((u, v))
            edges_set.add((4, 6))
            edges_set.add((1, 7))
            edges_set.discard((5, 12))
        else:
            # Random graph with 65 edges
            random.shuffle(all_pairs)
            edges_set = set(all_pairs[:65])
        
        edges_set = set(edges_set)
        current_score, triangles, adj_mask, deg = _compute_triangles_info(n, list(edges_set))
        
        if current_score < best_score:
            best_score = current_score
            best_edges = list(edges_set)
        
        # Simulated Annealing with targeted moves
        T_start = 3.0
        T_end = 0.01
        
        # Precompute non-edges for faster sampling
        non_edges = [e for e in all_pairs if e not in edges_set]
        
        for i in range(iterations_per_restart):
            if time.time() - start_time > DEADLINE - 10:
                break
                
            progress = i / iterations_per_restart
            T = T_start * math.pow(T_end / T_start, progress)
            
            # Targeted move: with 40% probability, remove an edge from a high-degree-sum triangle
            # and add a random non-edge
            if random.random() < 0.4 and current_score > 0 and triangles:
                # Pick a triangle with high sum
                triangles.sort(key=lambda x: x[1], reverse=True)
                top_k = min(8, len(triangles))
                tri, tri_sum = random.choice(triangles[:top_k])
                # Pick one of the three edges in this triangle to remove
                e1_candidates = [(tri[0], tri[1]), (tri[1], tri[2]), (tri[0], tri[2])]
                e1 = random.choice(e1_candidates)
                e1 = (min(e1[0], e1[1]), max(e1[0], e1[1]))
                if e1 not in edges_set:
                    e1 = random.choice(list(edges_set))
            else:
                e1 = random.choice(list(edges_set))
            
            # Get a random non-edge to add
            if non_edges:
                e2 = random.choice(non_edges)
            else:
                continue
            
            # Perform move
            new_edges_set = set(edges_set)
            new_edges_set.remove(e1)
            new_edges_set.add(e2)
            
            # Update adj_mask and deg incrementally
            new_adj_mask = adj_mask[:]
            new_deg = deg[:]
            
            # Remove e1
            u1, v1 = e1
            new_adj_mask[u1] &= ~(1 << v1)
            new_adj_mask[v1] &= ~(1 << u1)
            new_deg[u1] -= 1
            new_deg[v1] -= 1
            
            # Add e2
            u2, v2 = e2
            new_adj_mask[u2] |= (1 << v2)
            new_adj_mask[v2] |= (1 << u2)
            new_deg[u2] += 1
            new_deg[v2] += 1
            
            # Calculate new score using fast method
            new_score = _compute_score_fast(n, new_adj_mask, new_deg, list(new_edges_set))
            
            # Acceptance
            if new_score <= current_score:
                current_score = new_score
                edges_set = new_edges_set
                adj_mask = new_adj_mask
                deg = new_deg
                # Update non_edges
                non_edges.remove(e2)
                non_edges.append(e1)
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
                        adj_mask = new_adj_mask
                        deg = new_deg
                        non_edges.remove(e2)
                        non_edges.append(e1)
    
    # Final polish: local search with very low temperature on best found
    if best_edges is not None and time.time() - start_time < DEADLINE - 5:
        edges_set = set(best_edges)
        current_score = best_score
        polish_iters = 30000
        T_start = 0.5
        T_end = 0.01
        
        _, _, adj_mask, deg = _compute_triangles_info(n, list(edges_set))
        non_edges = [e for e in all_pairs if e not in edges_set]
        
        for i in range(polish_iters):
            if time.time() - start_time > DEADLINE - 3:
                break
                
            progress = i / polish_iters
            T = T_start * math.pow(T_end / T_start, progress)
            
            # Try swapping: remove one edge, add another
            e1 = random.choice(list(edges_set))
            
            if non_edges:
                e2 = random.choice(non_edges)
            else:
                continue
            
            new_edges_set = set(edges_set)
            new_edges_set.remove(e1)
            new_edges_set.add(e2)
            
            # Update adj_mask and deg incrementally
            new_adj_mask = adj_mask[:]
            new_deg = deg[:]
            
            u1, v1 = e1
            new_adj_mask[u1] &= ~(1 << v1)
            new_adj_mask[v1] &= ~(1 << u1)
            new_deg[u1] -= 1
            new_deg[v1] -= 1
            
            u2, v2 = e2
            new_adj_mask[u2] |= (1 << v2)
            new_adj_mask[v2] |= (1 << u2)
            new_deg[u2] += 1
            new_deg[v2] += 1
            
            new_score = _compute_score_fast(n, new_adj_mask, new_deg, list(new_edges_set))
            
            if new_score <= current_score:
                current_score = new_score
                edges_set = new_edges_set
                adj_mask = new_adj_mask
                deg = new_deg
                non_edges.remove(e2)
                non_edges.append(e1)
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
                        adj_mask = new_adj_mask
                        deg = new_deg
                        non_edges.remove(e2)
                        non_edges.append(e1)
    
    # Fallback
    if best_edges is None:
        edges_set = set()
        for u in range(8):
            for v in range(8, 16):
                edges_set.add((u, v))
        edges_set.add((0, 1))
        best_edges = list(edges_set)
        
    return {'n': n, 'edges': best_edges}
