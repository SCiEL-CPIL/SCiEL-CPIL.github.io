"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys
from collections import defaultdict

def construct():
    # Deadline guard: evaluator kills at 150s wall-clock. We aim to finish well before.
    start_time = time.time()
    DEADLINE = 140.0  # safety margin

    # Copy of the instance generator from the verifier
    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    specs = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))

    results = []
    for (n, p, seed) in specs:
        if time.time() > DEADLINE:
            # Fallback: return edge-as-path for remaining
            for remaining_spec in specs:
                edges = instance(*remaining_spec)
                results.append([list(e) for e in edges])
            return results

        edges = instance(n, p, seed)
        paths = _decompose(edges, n)
        results.append(paths)

    return results


def _decompose(edges, n, max_iterations=5000):
    """
    Greedy + local improvement path decomposition.
    Strategy:
      1. Start with each edge as its own path.
      2. Repeatedly try to merge two paths into one (by extending one with the other).
      3. Also try to find long paths via beam search and rebuild.
    
    A path is a list of vertices. Two paths can be merged if the end of one
    equals the start of the other (or vice versa), and the merged path is simple.
    """
    # Build adjacency
    adj = defaultdict(list)
    for (u, v) in edges:
        adj[u].append(v)
        adj[v].append(u)

    # Initial: each edge is a path
    paths = [list(e) for e in edges]
    used_edges = set(edges)

    # We'll work with a list of paths, each a list of vertices.
    # Edge set for a path [v0, v1, ..., vk] is {(v0,v1), (v1,v2), ..., (v[k-1],vk)} normalized.
    
    def path_edges(path):
        es = set()
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            es.add(tuple(sorted((u, v))))
        return es

    # Greedy merge: repeatedly try to merge paths
    # For efficiency, we track endpoints and try to extend.
    
    # Better approach: iterative improvement
    # Represent state as list of paths. Try all pairs to merge.
    
    def can_merge(p1, p2):
        """Check if p1 and p2 can be merged into a single simple path.
        Returns the merged path or None.
        """
        # Try: p1 + p2 (end of p1 == start of p2)
        if p1[-1] == p2[0]:
            merged = p1 + p2[1:]
            if len(set(merged)) == len(merged):
                # Check all edges exist
                for i in range(len(merged) - 1):
                    e = tuple(sorted((merged[i], merged[i + 1])))
                    if e not in used_edges:
                        return None
                return merged
        # Try: p2 + p1
        if p2[-1] == p1[0]:
            merged = p2 + p1[1:]
            if len(set(merged)) == len(merged):
                for i in range(len(merged) - 1):
                    e = tuple(sorted((merged[i], merged[i + 1])))
                    if e not in used_edges:
                        return None
                return merged
        # Try: p1 + reverse(p2)
        if p1[-1] == p2[-1]:
            merged = p1 + p2[-2::-1]
            if len(set(merged)) == len(merged):
                for i in range(len(merged) - 1):
                    e = tuple(sorted((merged[i], merged[i + 1])))
                    if e not in used_edges:
                        return None
                return merged
        # Try: p2 + reverse(p1)
        if p2[-1] == p1[-1]:
            merged = p2 + p1[-2::-1]
            if len(set(merged)) == len(merged):
                for i in range(len(merged) - 1):
                    e = tuple(sorted((merged[i], merged[i + 1])))
                    if e not in used_edges:
                        return None
                return merged
        return None

    # Iterative merging
    improved = True
    iteration = 0
    while improved and iteration < max_iterations:
        improved = False
        iteration += 1
        for i in range(len(paths)):
            for j in range(i + 1, len(paths)):
                merged = can_merge(paths[i], paths[j])
                if merged is not None:
                    paths[i] = merged
                    paths.pop(j)
                    improved = True
                    break
            if improved:
                break

    # Now try beam search to find long paths and rebuild
    # This is a more powerful approach: find a set of edge-disjoint paths covering all edges
    # using a different strategy.
    
    # Alternative: use a greedy longest-path-first approach
    # 1. Find a long path using beam search
    # 2. Remove its edges
    # 3. Repeat until no edges remain
    
    # But we already have a decomposition. Let's try to improve by:
    # - Finding a single long path in the current edge set
    # - Replacing the old paths with this long path + decomposition of remaining edges
    
    # Actually, let's try a cleaner approach: beam search for path cover
    
    def beam_search_path(edges_set, adj_map, beam_width=50, max_depth=None):
        """Find a long simple path using beam search."""
        if not edges_set:
            return []
        
        if max_depth is None:
            max_depth = len(edges_set) + 1
        
        # Start from each vertex
        # State: (path, set_of_used_edges)
        # To save memory, we track paths and their edge sets
        
        # Initialize beam with single-vertex paths
        beam = []
        for v in adj_map:
            beam.append(([v], set()))
        
        best_path = beam[0][0]
        
        for depth in range(max_depth):
            if time.time() > DEADLINE - 5:
                break
            new_beam = []
            seen_states = set()
            
            for path, used_e in beam:
                last = path[-1]
                for neighbor in adj_map[last]:
                    e = tuple(sorted((last, neighbor)))
                    if e in used_e:
                        continue
                    if neighbor in path:
                        continue
                    # Check edge exists
                    if e not in edges_set:
                        continue
                    new_path = path + [neighbor]
                    new_used = used_e | {e}
                    state_key = (tuple(new_path), frozenset(new_used))
                    # We'll use a simpler dedup: just the path tuple
                    path_key = tuple(new_path)
                    if path_key in seen_states:
                        continue
                    seen_states.add(path_key)
                    new_beam.append((new_path, new_used))
            
            if not new_beam:
                break
            
            # Sort by length (descending) and keep top beam_width
            new_beam.sort(key=lambda x: len(x[0]), reverse=True)
            beam = new_beam[:beam_width]
            
            if len(beam[0][0]) > len(best_path):
                best_path = beam[0][0]
            
            # If best path uses all edges, we're done
            if len(best_path) - 1 == len(edges_set):
                break
        
        return best_path

    # Try to improve using beam search based approach
    # Build a fresh decomposition using beam search
    def beam_decomposition(edges_list, adj_map, beam_width=100):
        """Decompose edges into paths using beam search for longest paths."""
        remaining = set(edges_list)
        paths = []
        
        while remaining:
            # Find longest path in remaining
            best = beam_search_path(remaining, adj_map, beam_width=beam_width)
            if len(best) <= 1:
                # Can't find a path, take any remaining edge
                e = next(iter(remaining))
                paths.append(list(e))
                remaining.discard(e)
                continue
            
            # Extract edges from best path
            path_edges = set()
            for i in range(len(best) - 1):
                pe = tuple(sorted((best[i], best[i + 1])))
                if pe in remaining:
                    path_edges.add(pe)
                else:
                    # Shouldn't happen, but handle gracefully
                    break
            
            if not path_edges:
                e = next(iter(remaining))
                paths.append(list(e))
                remaining.discard(e)
                continue
            
            paths.append(best)
            remaining -= path_edges
        
        return paths

    # Try beam decomposition and compare
    try:
        beam_paths = beam_decomposition(edges, adj, beam_width=80)
        if len(beam_paths) < len(paths):
            paths = beam_paths
    except Exception:
        pass

    return paths


# EVOLVE-BLOCK-END
