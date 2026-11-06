import time
import random
import heapq
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = 150.0  # Safety margin before 120s? No, task says 120s limit, evaluator kills at 150s. 
    # "Keep an EXPLICIT deadline guard: track elapsed time and return the best result found with a safety margin before the limit."
    # The limit is 120s. So I should aim to finish well before 120s. Let's set internal deadline to 110s to be safe.
    
    def get_graph(n, p, seed):
        rng = random.Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def solve_instance(n, p, seed):
        edges = get_graph(n, p, seed)
        if not edges:
            return []
        
        # Adjacency list for fast clique checks
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        uncovered_edges = set(edges)
        cliques = []
        
        # Sort edges to process in a somewhat deterministic order, 
        # maybe by sum of degrees to prioritize dense areas?
        # Actually, picking any uncovered edge is fine for the heuristic.
        # To improve quality, we can pick the edge involved in the most potential cliques,
        # but that's expensive. Let's just iterate.
        
        # We will repeatedly pick an uncovered edge, expand it to a clique,
        # and remove all edges of that clique from uncovered_edges.
        
        # To speed up, we can maintain a list of uncovered edges.
        # Since we are removing edges, a set is good for membership, but iteration order matters for heuristic quality.
        # Let's use a list and a set for O(1) removal checks if needed, or just set and pop.
        
        # A better heuristic: 
        # 1. Find a vertex u with max uncovered degree.
        # 2. Look at its uncovered neighbors.
        # 3. Find a clique in the subgraph of those neighbors.
        # 4. Add u + that clique to the partition.
        
        # Let's implement the "Max Uncovered Degree Vertex" heuristic.
        
        # Precompute initial uncovered degrees
        unc_deg = [0] * n
        for u, v in edges:
            unc_deg[u] += 1
            unc_deg[v] += 1
            
        # Use a heap for vertices? Or just scan? n is small (32, 64). Scanning is fast.
        # But we need to update degrees.
        
        while uncovered_edges:
            if time.time() > 110.0:
                # Fallback: return what we have + remaining edges as pairs
                # This might be invalid if we don't cover all edges.
                # The requirement is "every edge must lie in exactly one listed clique".
                # So we MUST cover all edges.
                # If we run out of time, we should just finish the current instance quickly 
                # by dumping all remaining edges as size-2 cliques.
                break
                
            # Find vertex with max uncovered degree
            max_deg = -1
            best_u = -1
            for u in range(n):
                if unc_deg[u] > max_deg:
                    max_deg = unc_deg[u]
                    best_u = u
            
            if best_u == -1 or max_deg == 0:
                # Should not happen if uncovered_edges is not empty
                # But just in case, pick any edge
                u, v = next(iter(uncovered_edges))
                cliques.append([u, v])
                uncovered_edges.remove((u, v))
                unc_deg[u] -= 1
                unc_deg[v] -= 1
                continue
            
            u = best_u
            
            # Get neighbors of u that are connected via an uncovered edge
            # Note: (u,v) is in uncovered_edges iff v in adj[u] and (u,v) not removed.
            # We maintain `uncovered_edges` set.
            
            candidates = []
            for v in adj[u]:
                if (u, v) in uncovered_edges or (v, u) in uncovered_edges:
                    candidates.append(v)
            
            if not candidates:
                # No uncovered edges from u? This contradicts unc_deg[u] > 0.
                # Maybe degree count is stale? 
                # Let's just pick any uncovered edge if this happens.
                e = next(iter(uncovered_edges))
                cliques.append(list(e))
                uncovered_edges.remove(e)
                # Update degrees for e
                x, y = e
                unc_deg[x] -= 1
                unc_deg[y] -= 1
                continue

            # We want to find a clique among `candidates` that is also a clique in G.
            # i.e., all vertices in the clique must be mutually adjacent.
            # And we want it to be as large as possible to cover many edges.
            
            # Greedy clique finding on the subgraph induced by candidates:
            # Sort candidates by their degree within the candidate set (or global degree) descending.
            # This is a standard greedy heuristic for Max Clique.
            
            # For each candidate v, count how many other candidates it is connected to.
            # This is O(|C|^2). |C| <= n. n=64. 64^2 = 4096. Very fast.
            
            cand_deg = {v: 0 for v in candidates}
            for i in range(len(candidates)):
                for j in range(i + 1, len(candidates)):
                    v1, v2 = candidates[i], candidates[j]
                    if v2 in adj[v1]:
                        cand_deg[v1] += 1
                        cand_deg[v2] += 1
            
            # Sort candidates by cand_deg descending
            candidates.sort(key=lambda x: cand_deg[x], reverse=True)
            
            clique = [u]
            # Try to add vertices to the clique
            for v in candidates:
                # Check if v is connected to all current members of clique (except u, which is already checked via candidates list)
                # Actually, v is in candidates, so (u,v) is an edge.
                # We need v to be connected to all other vertices in `clique` (which are from candidates).
                is_connected = True
                for w in clique:
                    if w == u:
                        continue
                    if w not in adj[v]:
                        is_connected = False
                        break
                if is_connected:
                    clique.append(v)
            
            # Now `clique` is a valid clique.
            cliques.append(clique)
            
            # Remove all edges within `clique` from uncovered_edges
            # And update unc_deg
            for i in range(len(clique)):
                for j in range(i + 1, len(clique)):
                    u1, u2 = clique[i], clique[j]
                    edge = (u1, u2) if u1 < u2 else (u2, u1)
                    if edge in uncovered_edges:
                        uncovered_edges.remove(edge)
                        unc_deg[u1] -= 1
                        unc_deg[u2] -= 1
        
        # If we broke out of the loop due to time, or if any edges remain (shouldn't happen in normal flow)
        # We must ensure ALL edges are covered.
        # The loop above removes edges. If it finishes, uncovered_edges is empty.
        # If it breaks due to time, we have uncovered_edges left.
        # We must add them as size-2 cliques to satisfy the constraint.
        # This is a valid partition, just not minimal.
        
        remaining = list(uncovered_edges)
        for u, v in remaining:
            cliques.append([u, v])
            
        return cliques

    # Solve both instances
    # Instance 1
    res1 = solve_instance(32, 0.6, 601)
    
    # Instance 2
    res2 = solve_instance(64, 0.55, 602)
    
    return [res1, res2]

# Verification helper (not part of the submitted solution, but for my own check if I could run it)
# The task asks for the code block only.

if __name__ == "__main__":
    # Quick test
    res = construct()
    print("Number of cliques in instance 1:", len(res[0]))
    print("Number of cliques in instance 2:", len(res[1]))
