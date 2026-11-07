import random
import time
import sys

def construct():
            start_time = time.time()
            deadline = start_time + 50.0
            n = 15
            best_edges = []
            
            # Helper functions
            def get_edges_list(edge_set):
                return [[u, v] for u, v in edge_set]
                
            def calculate_degrees(edge_set, n):
                deg = [0] * n
                for u, v in edge_set:
                    deg[u] += 1
                    deg[v] += 1
                return deg
                
            def is_valid(edge_set, n):
                deg = calculate_degrees(edge_set, n)
                # Build adjacency list for fast neighbor lookup
                adj = [set() for _ in range(n)]
                for u, v in edge_set:
                    adj[u].add(v)
                    adj[v].add(u)
                    
                # Check all 3-paths s-u-v-t
                # Iterate over edges (u, v)
                for u, v in edge_set:
                    # s is neighbor of u, not v
                    # t is neighbor of v, not u
                    # s != t
                    for s in adj[u]:
                        if s == v: continue
                        for t in adj[v]:
                            if t == u: continue
                            if s == t: continue
                            if deg[s] == deg[t]:
                                return False
                return True

            def find_violation(edge_set, n):
                deg = calculate_degrees(edge_set, n)
                adj = [set() for _ in range(n)]
                for u, v in edge_set:
                    adj[u].add(v)
                    adj[v].add(u)
                
                for u, v in edge_set:
                    for s in adj[u]:
                        if s == v: continue
                        for t in adj[v]:
                            if t == u: continue
                            if s == t: continue
                            if deg[s] == deg[t]:
                                return (s, u, v, t)
                return None

            def remove_random_critical_edge(edge_set, violation, n):
                s, u, v, t = violation
                candidates = set()
                # Edges in path
                candidates.add((min(u, v), max(u, v)))
                candidates.add((min(s, u), max(s, u)))
                candidates.add((min(v, t), max(v, t)))
                
                # Edges incident to s and t
                for u, v_e in edge_set:
                    if u == s or v_e == s:
                        candidates.add((min(u, v_e), max(u, v_e)))
                    if u == t or v_e == t:
                        candidates.add((min(u, v_e), max(u, v_e)))
                
                # Pick random
                cand_list = list(candidates)
                if not cand_list:
                    # Fallback: remove any edge
                    cand_list = list(edge_set)
                
                e = random.choice(cand_list)
                edge_set.remove(e)
                return e

            # Main Loop
            for seed in range(1000):
                if time.time() > deadline:
                    break
                
                random.seed(seed)
                # Initialize with complete graph
                current_edges = set()
                for i in range(n):
                    for j in range(i+1, n):
                        current_edges.add((i, j))
                
                # Removal Phase
                steps = 0
                max_steps = 200
                while steps < max_steps:
                    if time.time() > deadline:
                        break
                    violation = find_violation(current_edges, n)
                    if violation is None:
                        break
                    remove_random_critical_edge(current_edges, violation, n)
                    steps += 1
                
                # If we broke due to time, skip this run
                if time.time() > deadline:
                    break
                    
                # If still invalid after max steps, discard this attempt?
                # Or continue trying to fix?
                # Let's just discard if not valid.
                if not is_valid(current_edges, n):
                    continue
                
                # Addition Phase (Local Search)
                improved = True
                while improved:
                    if time.time() > deadline:
                        break
                    improved = False
                    # Get missing edges
                    all_edges = set()
                    for i in range(n):
                        for j in range(i+1, n):
                            all_edges.add((i, j))
                    missing = list(all_edges - current_edges)
                    random.shuffle(missing)
                    
                    for e in missing:
                        if time.time() > deadline:
                            break
                        current_edges.add(e)
                        if is_valid(current_edges, n):
                            improved = True
                        else:
                            current_edges.remove(e)
                            
                if len(current_edges) > len(best_edges):
                    best_edges = get_edges_list(current_edges)
                    
            # Return best found
            # Ensure types are correct
            return best_edges
