"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin
    
    n = 16
    # Edge list in lexicographic order
    edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    edge_idx = {e: idx for idx, e in enumerate(edges)}
    num_edges = len(edges)  # 120
    
    # Precompute all K4s
    all_k4s = list(itertools.combinations(range(n), 4))
    k4_edges = []
    for quad in all_k4s:
        a, b, c, d = quad
        k4_edges.append([
            edge_idx[(a,b)], edge_idx[(a,c)], edge_idx[(a,d)],
            edge_idx[(b,c)], edge_idx[(b,d)], edge_idx[(c,d)]
        ])
    
    def count_k4_colors(colors, k4e):
        """Return number of distinct colors in a K4."""
        return len(set(colors[e] for e in k4e))
    
    def is_valid(colors):
        """Check if every K4 has at least 5 distinct colors."""
        for k4e in k4_edges:
            if count_k4_colors(colors, k4e) < 5:
                return False
        return True
    
    def violation_count(colors):
        """Count K4s with fewer than 5 colors."""
        vc = 0
        for k4e in k4_edges:
            if count_k4_colors(colors, k4e) < 5:
                vc += 1
        return vc
    
    def try_local_search(colors, k, max_iters=5000):
        """Local search to fix violations. Returns (success, colors)."""
        colors = list(colors)
        vc = violation_count(colors)
        if vc == 0:
            return True, colors
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            # Find a violating K4
            viol_k4 = None
            for k4e in k4_edges:
                if count_k4_colors(colors, k4e) < 5:
                    viol_k4 = k4e
                    break
            if viol_k4 is None:
                return True, colors
            
            # Try to fix by changing one edge in this K4
            fixed = False
            for e in viol_k4:
                old_c = colors[e]
                # Try changing to a different color
                for new_c in range(k):
                    if new_c == old_c:
                        continue
                    colors[e] = new_c
                    # Check if this K4 is now valid
                    if count_k4_colors(colors, viol_k4) >= 5:
                        # Check we didn't create too many new violations
                        # Quick check: just accept and continue
                        fixed = True
                        break
                if fixed:
                    break
                colors[e] = old_c
            
            if not fixed:
                # Try random edge change
                e = random.randint(0, num_edges-1)
                old_c = colors[e]
                colors[e] = random.randint(0, k-1)
        
        vc = violation_count(colors)
        if vc == 0:
            return True, colors
        return False, colors
    
    def try_eliminate_color(colors, k):
        """Try to reduce number of colors by eliminating color k-1."""
        # Find all edges with color k-1
        to_recolor = [i for i in range(num_edges) if colors[i] == k-1]
        if not to_recolor:
            return True, colors, k-1
        
        for e in to_recolor:
            # Try to recolor this edge to another color without creating violations
            # that involve this edge
            # Get all K4s containing this edge
            old_c = colors[e]
            best_new = None
            for new_c in range(k-1):
                if new_c == old_c:
                    continue
                colors[e] = new_c
                # Check only K4s containing edge e
                ok = True
                for k4e in k4_edges:
                    if e in k4e and count_k4_colors(colors, k4e) < 5:
                        ok = False
                        break
                if ok:
                    best_new = new_c
                    break
                colors[e] = old_c
            
            if best_new is None:
                # Can't eliminate this color
                colors[e] = old_c
                return False, colors, k
            colors[e] = best_new
        
        return True, colors, k-1
    
    # Strategy: try different k values from low to high
    best_result = None
    best_k = 17
    
    # First, try the known valid solution as fallback
    fallback = [(i + j) % 17 for i in range(n) for j in range(i+1, n)]
    
    for k in range(13, 18):
        if time.time() > deadline - 10:
            break
        
        found = False
        for attempt in range(20):
            if time.time() > deadline - 5:
                break
            
            # Random initialization
            colors = [random.randint(0, k-1) for _ in range(num_edges)]
            
            # Local search
            success, colors = try_local_search(colors, k, max_iters=2000)
            if success:
                # Try to reduce colors further
                while k > 13:
                    if time.time() > deadline - 5:
                        break
                    ok, colors, k = try_eliminate_color(colors, k)
                    if not ok:
                        break
                found = True
                if k < best_k:
                    best_k = k
                    best_result = colors
                break
            elif attempt < 5:
                # Try with the fallback structure perturbed
                base = [(i + j) % k for i in range(n) for j in range(i+1, n)]
                # Perturb randomly
                colors = list(base)
                for _ in range(10):
                    e = random.randint(0, num_edges-1)
                    colors[e] = random.randint(0, k-1)
                success, colors = try_local_search(colors, k, max_iters=2000)
                if success:
                    while k > 13:
                        if time.time() > deadline - 5:
                            break
                        ok, colors, k = try_eliminate_color(colors, k)
                        if not ok:
                            break
                    found = True
                    if k < best_k:
                        best_k = k
                        best_result = colors
                    break
        
        if found and k <= 15:
            break
    
    if best_result is not None:
        # Final validation
        if is_valid(best_result):
            return best_result
    
    # Fallback to known valid solution
    return fallback
# EVOLVE-BLOCK-END
