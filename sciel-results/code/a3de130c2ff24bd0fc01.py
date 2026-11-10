# EVOLVE-BLOCK-START
import time
import random as _random

_T0 = time.time()
_DEADLINE = _T0 + 130.0  # wall-clock guard (evaluator kills at 150s)


def _random_graph(n, p, seed):
    """Reproduce the verifier's edge list exactly."""
    rng = _random.Random(seed)
    return [(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p]


def _generate_candidate_cliques(edges, n, rng, num_variants=3):
    """
    Generate a pool of candidate cliques using randomized greedy expansion.
    For each vertex, we try to build a clique starting from it.
    We do this multiple times with different random orderings to get diversity.
    """
    candidates = []
    if not edges:
        return candidates

    # Build adjacency list and bitmask for fast checks
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    adj_mask = [0] * n
    for u, v in edges:
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)

    vertices = list(range(n))
    
    for variant in range(num_variants):
        rng.shuffle(vertices)
        for start_v in vertices:
            if time.time() > _DEADLINE - 10:
                return candidates
            
            # Greedily expand clique starting at start_v
            C = [start_v]
            # Candidates must be neighbors of all current clique members
            # Start with neighbors of start_v
            current_cand = set(adj[start_v])
            
            while current_cand:
                # Pick a vertex from current_cand that has the highest degree within current_cand
                # This helps maximize the clique size
                best_u = None
                best_deg = -1
                # To speed up, we might not check all, but n is small (32/64)
                for u in current_cand:
                    # Count connections to other candidates
                    deg = 0
                    # Use bitmask for intersection
                    # adj_mask[u] & mask_of_current_cand
                    # But current_cand changes. Let's just loop for simplicity first, n is small.
                    for w in current_cand:
                        if u != w and w in adj[u]:
                            deg += 1
                    if deg > best_deg:
                        best_deg = deg
                        best_u = u
                
                if best_u is None:
                    break
                    
                C.append(best_u)
                
                # Update candidates: intersect with neighbors of best_u
                new_cand = set()
                for w in current_cand:
                    if w == best_u:
                        continue
                    if w in adj[best_u]:
                        new_cand.add(w)
                current_cand = new_cand
            
            if len(C) >= 2:
                # Ensure it's a clique (should be by construction, but safety check)
                # And add to candidates
                candidates.append(tuple(sorted(C)))

    # Deduplicate
    unique_candidates = list(set(candidates))
    return unique_candidates


def _solve_instance(edges, n, max_restarts=None):
    """
    Solve a single instance using Candidate Pool + Greedy Set Cover.
    """
    if not edges:
        return []

    if max_restarts is None:
        remaining_time = _DEADLINE - 5 - time.time()
        if remaining_time < 3:
            max_restarts = 1
        else:
            max_restarts = min(10, int(remaining_time / 5)) # Fewer restarts, more work per restart

    best_cliques = None
    best_count = float('inf')
    edge_set = set((min(u, v), max(u, v)) for u, v in edges)
    
    for restart in range(max_restarts):
        if time.time() > _DEADLINE - 5:
            break

        rng = _random.Random(2000 + restart)
        
        # Phase 1: Generate Candidates
        # We can vary the number of variants based on time
        variants = 5 if n <= 32 else 3
        candidates = _generate_candidate_cliques(edges, n, rng, num_variants=variants)
        
        if not candidates:
            # Fallback to simple edge cliques
            cliques = [[u, v] for u, v in edges]
            if len(cliques) < best_count:
                best_count = len(cliques)
                best_cliques = cliques
            continue

        # Phase 2: Greedy Set Cover
        # We want to select a subset of candidates such that every edge is covered exactly once.
        # Note: Candidates might overlap. We need to pick them such that they are disjoint in terms of edges.
        
        remaining_edges = set(edge_set)
        selected_cliques = []
        
        # Precompute edge sets for candidates for faster lookup?
        # Or just compute on the fly.
        
        while remaining_edges:
            if time.time() > _DEADLINE - 5:
                break
            
            best_c = None
            best_cover = 0
            
            for c in candidates:
                # Compute edges in c
                c_edges = set()
                k = len(c)
                for i in range(k):
                    for j in range(i+1, k):
                        e = (min(c[i], c[j]), max(c[i], c[j]))
                        if e in remaining_edges:
                            c_edges.add(e)
                
                # Only consider if it covers at least 1 remaining edge
                # And ideally, it doesn't cover edges that are NOT in remaining?
                # Actually, if a candidate covers an edge that is already covered, it's invalid for this step
                # because we need EXACTLY once.
                # So we must only pick candidates that are a subset of remaining_edges?
                # No, the candidate is a set of vertices. The edges induced by them must be a subset of remaining_edges.
                
                # Check if all edges induced by c are in remaining_edges
                valid = True
                for i in range(k):
                    for j in range(i+1, k):
                        e = (min(c[i], c[j]), max(c[i], c[j]))
                        if e not in remaining_edges:
                            valid = False
                            break
                    if not valid:
                        break
                
                if not valid:
                    continue
                
                cover_count = len(c_edges)
                if cover_count > best_cover:
                    best_cover = cover_count
                    best_c = c
            
            if best_c is None:
                # No valid candidate found. This shouldn't happen if candidates are comprehensive,
                # but if we missed some edges, we might have to fall back to 2-cliques.
                # Or maybe all remaining edges are isolated in the candidate pool.
                # We must cover them. Let's just take one edge as a 2-clique.
                e = next(iter(remaining_edges))
                best_c = tuple(sorted([e[0], e[1]]))
                best_cover = 1
            
            selected_cliques.append(list(best_c))
            
            # Remove edges covered by best_c from remaining
            k = len(best_c)
            for i in range(k):
                for j in range(i+1, k):
                    e = (min(best_c[i], best_c[j]), max(best_c[i], best_c[j]))
                    remaining_edges.discard(e)
            
            if time.time() > _DEADLINE - 5:
                break
        
        # Validate and Update Best
        if remaining_edges:
            # If we didn't finish in time or got stuck, this solution is invalid/incomplete
            # But we must return something. If we broke early, it's invalid.
            # Let's assume if we broke due to time, we discard this restart.
            continue

        # Final validation
        valid = True
        seen = set()
        for c in selected_cliques:
            k = len(c)
            if k < 2:
                valid = False
                break
            for i in range(k):
                for j in range(i+1, k):
                    e = (min(c[i], c[j]), max(c[i], c[j]))
                    if e not in edge_set or e in seen:
                        valid = False
                        break
                    seen.add(e)
            if not valid:
                break
        
        if valid and len(seen) == len(edge_set):
            if len(selected_cliques) < best_count:
                best_count = len(selected_cliques)
                best_cliques = selected_cliques

    if best_cliques is None:
        # Fallback
        best_cliques = [[u, v] for u, v in edges]

    return best_cliques


def construct():
    """
    Entry point: construct clique partitions for two fixed dense random graphs.
    Returns [cliques_for_graph1, cliques_for_graph2].
    """
    edges1 = _random_graph(32, 0.6, 601)
    edges2 = _random_graph(64, 0.55, 602)

    res1 = _solve_instance(edges1, 32)
    res2 = _solve_instance(edges2, 64)

    return [res1, res2]
# EVOLVE-BLOCK-END
