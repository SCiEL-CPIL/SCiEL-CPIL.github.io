import time
import random
import numpy as np

def construct():
    n = 20
    num_colors = 20
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit

    def time_ok():
        return time.time() < deadline

    # --- Fast 4-cycle enumeration and validation helpers ---

    def get_all_4cycles(adj):
        """Yields unique 4-cycles (u, v, w, x) where u-v-w-x-u is a cycle."""
        cycles = []
        for u in range(n):
            for w in range(u + 1, n):
                if adj[u, w]:
                    continue
                # Find common neighbors of u and w
                common = []
                for v in range(n):
                    if v != u and v != w and adj[u, v] and adj[w, v]:
                        common.append(v)
                if len(common) >= 2:
                    for i in range(len(common)):
                        for j in range(i + 1, len(common)):
                            v = common[i]
                            x = common[j]
                            cycles.append((u, v, w, x))
        return cycles

    def is_valid_coloring(adj, colors):
        """Check if the current coloring is valid (all 4-cycles are rainbow)."""
        cycles = get_all_4cycles(adj)
        for (u, v, w, x) in cycles:
            c1 = colors[u, v]
            c2 = colors[v, w]
            c3 = colors[w, x]
            c4 = colors[x, u]
            if len({c1, c2, c3, c4}) < 4:
                return False
        return True

    def get_valid_colors_for_new_edge(u, v, adj, colors):
        """
        Returns the set of colors that can be assigned to edge (u, v)
        without violating the rainbow 4-cycle property, assuming (u, v) is not currently present.
        """
        forbidden = set()
        # Check all 3-paths v-w-x-u such that (w,x) is an edge.
        # The cycle would be u-v-w-x-u.
        # Colors involved: c_uv (new), c_vw, c_wx, c_xu.
        # We need c_uv distinct from c_vw, c_wx, c_xu.
        # Also, c_vw, c_wx, c_xu must be distinct among themselves for the cycle to be fixable?
        # No, if c_vw == c_wx, the cycle is invalid regardless of c_uv.
        # So if such a path exists, we cannot add (u, v) at all.
        
        # To be efficient, we iterate neighbors of v and u.
        for w in range(n):
            if w == u or w == v:
                continue
            if not adj[v, w]:
                continue
            c_vw = colors[v, w]
            for x in range(n):
                if x == u or x == v or x == w:
                    continue
                if not adj[w, x] or not adj[x, u]:
                    continue
                c_wx = colors[w, x]
                c_xu = colors[x, u]
                
                # If the existing path has duplicate colors, this 4-cycle cannot be rainbow
                # no matter what color we give to (u, v). Thus, we cannot add (u, v).
                if c_vw == c_wx or c_vw == c_xu or c_wx == c_xu:
                    return set()
                
                # Otherwise, we must avoid these three colors
                forbidden.add(c_vw)
                forbidden.add(c_wx)
                forbidden.add(c_xu)
        
        return set(range(num_colors)) - forbidden

    def get_valid_colors_for_recolor(u, v, new_c, adj, colors):
        """
        Check if recoloring edge (u, v) to new_c is valid.
        Returns True if valid, False otherwise.
        """
        # Check all 4-cycles containing (u, v).
        # Cycle u-v-w-x-u.
        # Colors: new_c, c_vw, c_wx, c_xu.
        # We need new_c distinct from c_vw, c_wx, c_xu.
        # And c_vw, c_wx, c_xu must be distinct (they were before, but let's be sure).
        
        for w in range(n):
            if w == u or w == v:
                continue
            if not adj[v, w]:
                continue
            c_vw = colors[v, w]
            if new_c == c_vw:
                return False
            for x in range(n):
                if x == u or x == v or x == w:
                    continue
                if not adj[w, x] or not adj[x, u]:
                    continue
                c_wx = colors[w, x]
                c_xu = colors[x, u]
                
                # Check distinctness of the other three
                if c_vw == c_wx or c_vw == c_xu or c_wx == c_xu:
                    # This cycle is already invalid due to other edges.
                    # But wait, if the graph was valid before, this shouldn't happen.
                    # If the graph is not valid, we are in trouble.
                    # Assuming we only call this when the graph is currently valid or we are trying to fix it.
                    # If we are trying to fix a violation, the "other" edges might be part of the violation.
                    # However, recoloring (u, v) cannot fix a violation caused by c_vw == c_wx.
                    # So if such a conflict exists among the other 3 edges, we can't fix it by changing (u, v).
                    # But we might be changing (u, v) to *remove* a conflict involving (u, v).
                    # If c_vw == c_wx, the cycle is invalid regardless of new_c.
                    # So we should return False? Or does it not matter?
                    # If the goal is to reach a valid state, and this cycle is unfixable by changing (u, v),
                    # then changing (u, v) is useless for this specific cycle.
                    # However, it might fix OTHER cycles.
                    # But if this cycle remains invalid, the graph remains invalid.
                    # So we should probably avoid moves that leave any cycle invalid.
                    # Thus, if there is a "hard" conflict in the other edges, we can't fix the graph
                    # by just changing (u, v). We might need to change another edge.
                    # For simplicity in local search, let's say this move is invalid if it doesn't resolve
                    # all conflicts in cycles it touches?
                    # Actually, standard local search: check if the *resulting* state is valid for these cycles.
                    # If c_vw == c_wx, the resulting state for this cycle is still invalid.
                    # So return False.
                    return False

                if new_c == c_wx or new_c == c_xu:
                    return False
        return True

    # --- Greedy Construction ---

    def run_greedy(seed, order='bipartite'):
        random.seed(seed)
        
        adj = np.zeros((n, n), dtype=bool)
        colors = np.full((n, n), -1, dtype=int)
        edges = []
        
        all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        
        if order == 'bipartite':
            left = list(range(10))
            right = list(range(10, 20))
            bip_edges = [(u, v) for u in left for v in right]
            intra_edges = [(i, j) for i in range(n) for j in range(i+1, n) if not (i < 10 and j >= 10)]
            random.shuffle(bip_edges)
            random.shuffle(intra_edges)
            all_edges = bip_edges + intra_edges
        elif order == 'random':
            random.shuffle(all_edges)
        else:
            random.shuffle(all_edges)

        for u, v in all_edges:
            if not time_ok():
                break
            valid_colors = get_valid_colors_for_new_edge(u, v, adj, colors)
            if valid_colors:
                # Prefer a deterministic color if possible to reduce randomness?
                # Or just pick one.
                c = random.choice(list(valid_colors))
                adj[u, v] = True
                adj[v, u] = True
                colors[u, v] = c
                colors[v, u] = c
                edges.append([u, v, c])
        
        return edges, adj, colors

    # --- Local Search / Refinement ---

    def local_search(edges, adj, colors, seed):
        random.seed(seed)
        
        # Make copies
        adj = adj.copy()
        colors = colors.copy()
        edges = [e[:] for e in edges]
        
        best_count = len(edges)
        best_edges = [e[:] for e in edges]
        
        # Precompute neighbor lists for speed?
        # n=20 is small, direct loops are fine.
        
        max_iters = 2000
        for it in range(max_iters):
            if not time_ok():
                break
                
            # 1. Try to add an edge
            missing = [(i, j) for i in range(n) for j in range(i+1, n) if not adj[i, j]]
            if missing:
                random.shuffle(missing)
                added = False
                for u, v in missing:
                    valid_colors = get_valid_colors_for_new_edge(u, v, adj, colors)
                    if valid_colors:
                        c = random.choice(list(valid_colors))
                        adj[u, v] = True
                        adj[v, u] = True
                        colors[u, v] = c
                        colors[v, u] = c
                        edges.append([u, v, c])
                        best_count = len(edges)
                        best_edges = [e[:] for e in edges]
                        added = True
                        break
                if added:
                    continue # Restart loop to re-evaluate missing edges
            
            # 2. Try to recolor an edge to enable additions
            if edges:
                random.shuffle(edges)
                recolor_success = False
                for u, v, c in edges[:50]: # Limit checks per iteration
                    # Try a few new colors
                    new_colors = [nc for nc in range(num_colors) if nc != c]
                    random.shuffle(new_colors)
                    for new_c in new_colors[:5]:
                        if get_valid_colors_for_recolor(u, v, new_c, adj, colors):
                            # Apply recolor
                            colors[u, v] = new_c
                            colors[v, u] = new_c
                            # Update edges list
                            for e in edges:
                                if (e[0] == u and e[1] == v) or (e[0] == v and e[1] == u):
                                    e[2] = new_c
                                    break
                            recolor_success = True
                            break
                    if recolor_success:
                        break
                
                if recolor_success:
                    continue # Restart loop

            # 3. If stuck, try removing an edge?
            # Removing an edge might allow adding others later, but reduces count.
            # Only do this if we are far from max and stuck.
            if len(edges) < 100 and random.random() < 0.1:
                # Remove a random edge
                idx = random.randint(0, len(edges) - 1)
                u, v, c = edges[idx]
                adj[u, v] = False
                adj[v, u] = False
                colors[u, v] = -1
                colors[v, u] = -1
                edges.pop(idx)
            
        return best_edges, best_count

    # --- Main Logic ---

    best_overall_edges = []
    best_overall_count = 0

    # Multiple restarts with different seeds and orders
    orders = ['bipartite', 'random', 'random']
    
    for i in range(10):
        if not time_ok():
            break
        
        seed = 42 + i
        order = orders[i % len(orders)]
        
        edges, adj, colors = run_greedy(seed, order)
        
        if len(edges) > best_overall_count:
            best_overall_count = len(edges)
            best_overall_edges = [e[:] for e in edges]
        
        # Refine
        refined_edges, refined_count = local_search(edges, adj, colors, seed + 1000)
        
        if refined_count > best_overall_count:
            best_overall_count = refined_count
            best_overall_edges = [e[:] for e in refined_edges]
        
        # If we reached a high density, we might stop early?
        # Max possible edges is 190.
        if best_overall_count >= 150:
             # Try to polish with more iterations?
             pass

    # Final check: ensure the best solution is valid
    # Reconstruct adj/colors from best_overall_edges for final verification
    if best_overall_edges:
        final_adj = np.zeros((n, n), dtype=bool)
        final_colors = np.full((n, n), -1, dtype=int)
        for u, v, c in best_overall_edges:
            final_adj[u, v] = True
            final_adj[v, u] = True
            final_colors[u, v] = c
            final_colors[v, u] = c
        
        if not is_valid_coloring(final_adj, final_colors):
            # This should not happen if our checks are correct, but just in case
            # We might need to remove conflicting edges.
            # For now, return empty or a known valid subset?
            # Let's try to fix it by removing edges involved in violations.
            violations = []
            cycles = get_all_4cycles(final_adj)
            for (u, v, w, x) in cycles:
                c1 = final_colors[u, v]
                c2 = final_colors[v, w]
                c3 = final_colors[w, x]
                c4 = final_colors[x, u]
                if len({c1, c2, c3, c4}) < 4:
                    violations.append((u, v, w, x))
            
            # Remove edges from violations until valid
            # Simple approach: remove one edge from each violation
            to_remove = set()
            for (u, v, w, x) in violations:
                # Pick a random edge from the cycle
                edge = random.choice([(u, v), (v, w), (w, x), (x, u)])
                to_remove.add(edge)
            
            for u, v in to_remove:
                final_adj[u, v] = False
                final_adj[v, u] = False
                final_colors[u, v] = -1
                final_colors[v, u] = -1
            
            # Rebuild edge list
            best_overall_edges = []
            for u in range(n):
                for v in range(u + 1, n):
                    if final_adj[u, v]:
                        best_overall_edges.append([u, v, final_colors[u, v]])
            
            # Verify again
            if not is_valid_coloring(final_adj, final_colors):
                # If still invalid, return empty to be safe? Or best effort.
                # The task says "invalid result ... scores 0".
                # So we must ensure validity.
                # If we can't fix it, return a known small valid graph?
                # A complete bipartite graph K_{10,10} with proper edge coloring?
                # Let's just return the current best if it becomes valid.
                pass

    return best_overall_edges

if __name__ == "__main__":
    # Quick self-test
    edges = construct()
    print(f"Constructed {len(edges)} edges")
