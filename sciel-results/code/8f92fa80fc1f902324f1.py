"""Strong edge colourings via Iterative Maximum Induced Matching Extraction."""
# EVOLVE-BLOCK-START
import time
import random
import sys
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    def instance(n, d, seed):
        rng = random.Random(seed)
        degree, edges = [0] * n, set()
        for _ in range(4 * n * d):
            u, v = rng.randrange(n), rng.randrange(n)
            e = (min(u, v), max(u, v))
            if u != v and degree[u] < d and degree[v] < d and e not in edges:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        return sorted(edges)

    def build_structures(edges, n):
        """Build vertex adjacency and edge index mapping."""
        m = len(edges)
        edge_index = {e: i for i, e in enumerate(edges)}
        vertex_adj = defaultdict(set)
        for u, v in edges:
            vertex_adj[u].add(v)
            vertex_adj[v].add(u)
        return edge_index, vertex_adj, m

    def get_conflict_mask(edge_i, edges, edge_index, vertex_adj):
        """Get bitmask of edges conflicting with edge_i."""
        u, v = edges[edge_i]
        mask = 0
        # All edges sharing u or v
        for x in (u, v):
            for y in vertex_adj[x]:
                e2 = (min(x, y), max(x, y))
                if e2 in edge_index:
                    j = edge_index[e2]
                    if j != edge_i:
                        mask |= (1 << j)
        # All edges joined by an edge (distance 2 in line graph sense)
        for x in (u, v):
            for y in vertex_adj[x]:
                if y == u or y == v:
                    continue
                for z in vertex_adj[y]:
                    if z == x:
                        continue
                    e2 = (min(y, z), max(y, z))
                    if e2 in edge_index:
                        j = edge_index[e2]
                        if j != edge_i:
                            mask |= (1 << j)
        return mask

    def build_conflict_masks(edges, n):
        """Precompute conflict masks for all edges."""
        m = len(edges)
        edge_index, vertex_adj, _ = build_structures(edges, n)
        conflicts = [0] * m
        for i in range(m):
            conflicts[i] = get_conflict_mask(i, edges, edge_index, vertex_adj)
        return conflicts

    def is_induced_matching(candidate_edges, edges, conflicts):
        """Check if candidate_edges form an induced matching."""
        for i_idx, e1 in enumerate(candidate_edges):
            for e2 in candidate_edges[i_idx+1:]:
                if conflicts[e1] & (1 << e2):
                    return False
        return True

    def greedy_induced_matching(remaining_mask, edges, conflicts, n_edges, rng):
        """Find a maximal induced matching using greedy with random ordering."""
        # Get list of remaining edges
        remaining = [i for i in range(n_edges) if (remaining_mask >> i) & 1]
        if not remaining:
            return [], remaining_mask
        
        rng.shuffle(remaining)
        matching = []
        matched_vertices = set()
        
        for e_idx in remaining:
            u, v = edges[e_idx]
            if u in matched_vertices or v in matched_vertices:
                continue
            # Check if this edge conflicts with any edge in current matching
            can_add = True
            for m_idx in matching:
                if conflicts[e_idx] & (1 << m_idx):
                    can_add = False
                    break
            if can_add:
                matching.append(e_idx)
                matched_vertices.add(u)
                matched_vertices.add(v)
        
        # Update remaining mask
        new_remaining = remaining_mask
        for e_idx in matching:
            new_remaining &= ~(1 << e_idx)
            # Also remove all edges that conflict with any edge in matching
            for m_idx in matching:
                new_remaining &= ~conflicts[e_idx]
        
        return matching, new_remaining

    def local_improve_matching(matching, remaining_mask, edges, conflicts, n_edges, rng, time_limit):
        """Try to improve the matching by swapping edges."""
        if not matching:
            return matching, remaining_mask
        
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            # Try to add edges that were excluded
            for e_idx in range(n_edges):
                if (remaining_mask >> e_idx) & 1:
                    continue
                # Check if we can add this edge by removing one from matching
                u, v = edges[e_idx]
                # Find which matching edges block this
                blockers = []
                for m_idx in matching:
                    if conflicts[e_idx] & (1 << m_idx):
                        blockers.append(m_idx)
                
                # Try removing each blocker and see if we can still maintain induced matching
                for b_idx in blockers:
                    new_matching = [m for m in matching if m != b_idx]
                    new_matching.append(e_idx)
                    if is_induced_matching(new_matching, edges, conflicts):
                        matching = new_matching
                        # Update remaining mask
                        remaining_mask |= (1 << b_idx)
                        remaining_mask &= ~(1 << e_idx)
                        # Remove conflicts of new matching
                        for m_idx in matching:
                            remaining_mask &= ~conflicts[m_idx]
                        improved = True
                        break
                if improved:
                    break
        
        return matching, remaining_mask

    def solve_instance(edges, n, d, seed, time_limit):
        m = len(edges)
        if m == 0:
            return []
        
        conflicts = build_conflict_masks(edges, n)
        
        best_colours = None
        best_count = float('inf')
        
        rng = random.Random(seed)
        
        max_restarts = 30
        restart_count = 0
        
        while restart_count < max_restarts and time.time() < time_limit:
            # Iteratively extract induced matchings
            colours = [-1] * m
            remaining_mask = (1 << m) - 1
            color_count = 0
            
            while remaining_mask and time.time() < time_limit:
                # Extract an induced matching
                matching, remaining_mask = greedy_induced_matching(
                    remaining_mask, edges, conflicts, m, rng
                )
                
                if not matching:
                    # Fallback: assign next color to one edge
                    for i in range(m):
                        if (remaining_mask >> i) & 1:
                            colours[i] = color_count
                            remaining_mask &= ~(1 << i)
                            for j in range(m):
                                if conflicts[i] & (1 << j):
                                    remaining_mask &= ~(1 << j)
                            break
                else:
                    for e_idx in matching:
                        colours[e_idx] = color_count
                
                color_count += 1
            
            # Ensure all edges are colored
            for i in range(m):
                if colours[i] == -1:
                    colours[i] = color_count
                    color_count += 1
            
            current_count = max(colours) + 1 if m > 0 else 0
            
            if current_count < best_count:
                best_count = current_count
                best_colours = colours[:]
            
            restart_count += 1
        
        if best_colours is None:
            # Fallback greedy
            best_colours = [0] * m
            for i in range(m):
                blocked = set()
                for j in range(m):
                    if j != i and (conflicts[i] >> j) & 1:
                        blocked.add(best_colours[j])
                c = 0
                while c in blocked:
                    c += 1
                best_colours[i] = c
        
        return best_colours

    results = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        n, d, seed = spec
        edges = instance(n, d, seed)
        
        # Allocate time per instance
        inst_start = time.time()
        inst_limit = min(inst_start + 45.0, deadline - 5.0)
        
        colours = solve_instance(edges, n, d, seed, inst_limit)
        results.append(colours)
        
        if time.time() > deadline:
            break

    return results
# EVOLVE-BLOCK-END
