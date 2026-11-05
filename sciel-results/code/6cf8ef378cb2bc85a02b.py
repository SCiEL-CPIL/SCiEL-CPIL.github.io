import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = 145.0  # safety margin
    
    n = 16
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    m = len(edges)  # 120
    
    # Precompute K4 edge indices
    k4s = []
    for quad in combinations(range(n), 4):
        a, b, c, d = quad
        k4_edges = [(a,b), (a,c), (a,d), (b,c), (b,d), (c,d)]
        k4s.append(k4_edges)
    
    # Edge index map
    edge_idx = {}
    for idx, e in enumerate(edges):
        edge_idx[e] = idx
        
    # Precompute K4 edge index arrays for fast access
    k4_edge_idxs = []
    for k4e in k4s:
        k4_edge_idxs.append([edge_idx[e] for e in k4e])

    def count_colors(coloring):
        return len(set(coloring.tolist()))
    
    def check_valid(coloring):
        """Check if every K4 has at least 5 distinct colors."""
        for idxs in k4_edge_idxs:
            # Get the 6 colors
            c0 = coloring[idxs[0]]
            c1 = coloring[idxs[1]]
            c2 = coloring[idxs[2]]
            c3 = coloring[idxs[3]]
            c4 = coloring[idxs[4]]
            c5 = coloring[idxs[5]]
            
            # Check if at least 5 distinct
            # A set of 6 elements has < 5 distinct iff it has <= 4 distinct
            # i.e., there are duplicates reducing the count.
            # Faster check: 
            if c0 == c1 and c0 == c2 and c0 == c3 and c0 == c4 and c0 == c5:
                return False # 1 color
            # ... this is getting complicated. A simple set check is fast enough for 1820 checks.
            if len({c0, c1, c2, c3, c4, c5}) < 5:
                return False
        return True

    def count_violations(coloring):
        """Count K4s with fewer than 5 distinct colors."""
        violations = 0
        for idxs in k4_edge_idxs:
            c0 = coloring[idxs[0]]
            c1 = coloring[idxs[1]]
            c2 = coloring[idxs[2]]
            c3 = coloring[idxs[3]]
            c4 = coloring[idxs[4]]
            c5 = coloring[idxs[5]]
            if len({c0, c1, c2, c3, c4, c5}) < 5:
                violations += 1
        return violations

    best_coloring = None
    best_colors = 120  # worst case
    
    random.seed(42)
    
    # Strategy: Try structured colorings based on vertex labels
    # color(i, j) = f(label[i], label[j]) % mod
    
    # Functions to test
    def func_sum(l1, l2, mod): return (l1 + l2) % mod
    def func_xor(l1, l2, mod): return (l1 ^ l2) % mod
    def func_diff(l1, l2, mod): return (l1 - l2) % mod
    def func_prod(l1, l2, mod): return (l1 * l2) % mod
    
    functions = [func_sum, func_xor, func_diff, func_prod]
    func_names = ['sum', 'xor', 'diff', 'prod']
    
    # We prioritize smaller moduli to minimize color count
    # Target: find valid coloring with mod 13, 14, 15
    for mod in [13, 14, 15, 16, 17]:
        if time.time() > deadline - 10:
            break
            
        for f_idx, f in enumerate(functions):
            if time.time() > deadline - 10:
                break
                
            # Try multiple random labelings
            max_trials = 500 if mod <= 15 else 100
            for trial in range(max_trials):
                if time.time() > deadline - 5:
                    break
                    
                # Assign random labels to vertices
                labels = [random.randint(0, mod - 1) for _ in range(n)]
                
                # Derive edge colors
                coloring = np.zeros(m, dtype=int)
                for idx, (i, j) in enumerate(edges):
                    coloring[idx] = f(labels[i], labels[j], mod)
                
                # Check validity
                if check_valid(coloring):
                    n_colors = count_colors(coloring)
                    if n_colors < best_colors:
                        best_colors = n_colors
                        best_coloring = coloring.copy()
                        
                        # If we found a very good solution (e.g., 13 or 14), we can stop early
                        # but let's try to improve it slightly or just return if it's optimal-ish
                        if best_colors <= 13:
                            # Can't get better than 13 (lower bound)
                            # Return immediately? No, let's ensure it's robust.
                            pass
                    else:
                        # Even if not better, we have a valid one. 
                        # But we want to minimize.
                        pass
                else:
                    # Local search: try to fix violations by changing one vertex label
                    # This is a simple hill climbing step
                    for _ in range(20):
                        if time.time() > deadline - 5:
                            break
                        v = random.randint(0, n - 1)
                        old_label = labels[v]
                        new_label = random.randint(0, mod - 1)
                        if new_label == old_label:
                            continue
                        labels[v] = new_label
                        
                        # Recompute colors for edges incident to v
                        for idx, (i, j) in enumerate(edges):
                            if i == v or j == v:
                                coloring[idx] = f(labels[i], labels[j], mod)
                                
                        if check_valid(coloring):
                            n_colors = count_colors(coloring)
                            if n_colors < best_colors:
                                best_colors = n_colors
                                best_coloring = coloring.copy()
                            break
                        else:
                            labels[v] = old_label
                            # Recompute back? Or just continue loop. 
                            # For simplicity, revert and recompute in next iteration if needed.
                            # Actually, we should revert coloring too if we don't accept.
                            # But the loop re-computes incident edges for the NEW label.
                            # If we reject, we need to restore old colors for incident edges.
                            # To save code, let's just recompute the whole coloring if we reject? 
                            # No, that's slow. 
                            # Let's just accept the risk of slight inefficiency or recompute incident edges back.
                            for idx, (i, j) in enumerate(edges):
                                if i == v or j == v:
                                    coloring[idx] = f(labels[i], labels[j], mod)
                                    # Wait, labels[v] is back to old_label, so this restores it.
                                    # But we only do this if we break or continue.
                                    # If we continue, we need to restore.
                                    # Let's optimize: only restore if we don't break.
                            # Actually, the code above restores labels[v].
                            # We need to restore coloring for incident edges.
                            for idx, (i, j) in enumerate(edges):
                                if i == v or j == v:
                                    coloring[idx] = f(labels[i], labels[j], mod)
                            # This is getting messy. Let's simplify: 
                            # Just restart the trial if local search fails.
                            # Or, better yet, do a proper local search.
                        
                        # To keep it simple and robust, if local search doesn't find a solution in 20 steps, 
                        # we just try a new random labeling.

    # If no structured solution found, fall back to the seed's approach or a simple valid one
    if best_coloring is None:
        # The (i+j)%17 is a known valid coloring for many such problems
        coloring = np.array([(i + j) % 17 for i in range(n) for j in range(i + 1, n)], dtype=int)
        if check_valid(coloring):
            best_coloring = coloring
            best_colors = count_colors(coloring)
        else:
            # Last resort: distinct colors for all edges
            coloring = np.arange(m, dtype=int)
            best_coloring = coloring
            best_colors = m

    # Final Polish: Try to reduce colors on the best coloring found
    # If best_colors > 13, try to merge colors
    if best_colors > 13:
        coloring = best_coloring.copy()
        current_colors = best_colors
        
        # Simple local search: try to recolor an edge to an existing color
        # to reduce the total number of unique colors
        for _ in range(5000):
            if time.time() > deadline - 2:
                break
            
            # Find the color with the highest frequency? 
            # Or just pick a random edge and try to change it to a color 
            # that is already present, if it doesn't violate constraints.
            
            # To reduce color count, we need to change an edge of color C 
            # to a color D such that D is already used elsewhere, 
            # and C is not used elsewhere (so C disappears).
            
            color_counts = {}
            for c in coloring:
                color_counts[c] = color_counts.get(c, 0) + 1
            
            # Find colors that appear exactly once
            singleton_colors = [c for c, count in color_counts.items() if count == 1]
            
            if not singleton_colors:
                break
                
            target_color = random.choice(singleton_colors)
            # Find the edge with this color
            target_idx = np.where(coloring == target_color)[0][0]
            
            # Try to change it to another existing color
            other_colors = [c for c in color_counts.keys() if c != target_color]
            random.shuffle(other_colors)
            
            for new_color in other_colors:
                old_color = coloring[target_idx]
                coloring[target_idx] = new_color
                
                if check_valid(coloring):
                    new_count = count_colors(coloring)
                    if new_count < current_colors:
                        current_colors = new_count
                        best_colors = new_count
                        best_coloring = coloring.copy()
                        break
                    else:
                        coloring[target_idx] = old_color
                        break
                else:
                    coloring[target_idx] = old_color
                    # If one fails, try next? Or break? 
                    # Trying all is expensive. Let's break and try another singleton later.
                    break

    # Return the best coloring found
    result = [int(c) for c in best_coloring]
    # Remap colors to 0..K-1
    unique_colors = sorted(set(result))
    color_map = {c: i for i, c in enumerate(unique_colors)}
    result = [color_map[c] for c in result]
    
    return result
