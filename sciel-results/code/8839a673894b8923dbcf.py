import time
import random
import sys
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin for safety

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

    def solve_instance(edges, adj):
        m = len(edges)
        if m == 0:
            return []
        
        # Build incompatibility graph
        incompatible = [set() for _ in range(m)]
        
        for i in range(m):
            u1, v1 = edges[i]
            for j in range(i + 1, m):
                u2, v2 = edges[j]
                if u1 == u2 or u1 == v2 or v1 == u2 or v1 == v2:
                    incompatible[i].add(j)
                    incompatible[j].add(i)
                    continue
                if u2 in adj[u1] or v2 in adj[u1] or u2 in adj[v1] or v2 in adj[v1]:
                    incompatible[i].add(j)
                    incompatible[j].add(i)
        
        # Initial Greedy Coloring
        degrees = [len(incompatible[i]) for i in range(m)]
        order = sorted(range(m), key=lambda x: -degrees[x])
        
        colors = [-1] * m
        current_color = 0
        
        for v in order:
            used_colors = set()
            for u in incompatible[v]:
                if colors[u] != -1:
                    used_colors.add(colors[u])
            
            c = 0
            while c in used_colors:
                c += 1
            colors[v] = c
            if c + 1 > current_color:
                current_color = c + 1
        
        # Local Search: Iterative Color Merging
        # We try to merge color class 'target' into 'source'.
        # If conflicts arise, we try to move conflicting edges to other colors.
        
        def try_merge(source, target):
            # Returns True if merge was successful, modifying colors and current_color
            nonlocal current_color
            
            # Edges in target class
            target_edges = [i for i in range(m) if colors[i] == target]
            if not target_edges:
                return True # Nothing to merge
            
            # Check if we can simply move all target_edges to source
            # Conflicts: edges in target_edges that are incompatible with edges in source class
            source_edges = set(i for i in range(m) if colors[i] == source)
            
            conflicts = []
            for i in target_edges:
                for j in incompatible[i]:
                    if j in source_edges:
                        conflicts.append(i)
                        break
            
            if not conflicts:
                # No conflicts, just recolor
                for i in target_edges:
                    colors[i] = source
                # If target was the highest color, we might reduce current_color
                if target == current_color - 1:
                    # Check if target is still used
                    for i in range(m):
                        if colors[i] == target:
                            break
                    else:
                        current_color -= 1
                return True
            
            # We have conflicts. Try to resolve by moving conflicting edges to other colors.
            # This is a heuristic. We only attempt if the number of conflicts is small.
            if len(conflicts) > 5:
                return False

            # Save state
            old_colors = colors[:]
            old_current_color = current_color
            
            # Move all target edges to source
            for i in target_edges:
                colors[i] = source
            
            # Now resolve conflicts for the specific edges that caused issues
            # Actually, the conflicts are edges in target_edges that now clash with source_edges.
            # We need to move these specific edges to *other* colors.
            
            resolved = True
            for i in conflicts:
                # Find a valid color for i, excluding source (since it clashes)
                # and excluding target (which is being emptied/merged)
                valid_colors = set(range(current_color + 1)) # Allow new color
                # Remove colors used by incompatible neighbors
                for j in incompatible[i]:
                    if colors[j] != -1:
                        valid_colors.discard(colors[j])
                
                if not valid_colors:
                    resolved = False
                    break
                
                # Pick the smallest valid color to keep palette compact
                new_c = min(valid_colors)
                colors[i] = new_c
                
                # If we used a new color, update current_color
                if new_c >= current_color:
                    current_color = new_c + 1
            
            if resolved:
                # Check if we actually reduced colors or kept same
                # The merge itself reduces the "logical" classes by 1, but we might have introduced new colors.
                # We accept if current_color <= old_current_color
                if current_color <= old_current_color:
                    # Success
                    # Compact colors? Not strictly necessary for count, but good for hygiene.
                    # If target was the max color and is now empty, current_color might have dropped.
                    # We need to ensure current_color reflects the max used color.
                    max_used = -1
                    for i in range(m):
                        if colors[i] > max_used:
                            max_used = colors[i]
                    current_color = max_used + 1
                    
                    if current_color <= old_current_color:
                        return True
                    else:
                        # It increased, revert
                        colors[:] = old_colors
                        current_color = old_current_color
                        return False
                else:
                    # It increased, revert
                    colors[:] = old_colors
                    current_color = old_current_color
                    return False
            else:
                # Failed to resolve
                colors[:] = old_colors
                current_color = old_current_color
                return False

        # Main loop
        rng = random.Random(42)
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            # Try all pairs of colors? Or random pairs?
            # Random pairs is faster to explore.
            # We try to merge the color with the fewest edges into another.
            # Or just random pairs.
            
            # Strategy: Pick a random color 'target' to eliminate, and a random 'source' to merge into.
            # To be effective, we should try to eliminate colors that are "small" or "easy".
            
            # Get list of used colors
            used_colors = set(colors)
            if len(used_colors) == 1:
                break
                
            # Randomly select target and source
            # Bias towards eliminating colors with fewer edges?
            color_counts = defaultdict(int)
            for c in colors:
                color_counts[c] += 1
            
            # Sort colors by count ascending, pick from the bottom half for target
            sorted_colors = sorted(color_counts.keys(), key=lambda x: color_counts[x])
            # Only try to eliminate colors that are not the only one
            candidates = [c for c in sorted_colors if c != sorted_colors[-1] or len(sorted_colors) > 1]
            if not candidates:
                break
            
            target = rng.choice(candidates)
            source = rng.choice([c for c in used_colors if c != target])
            
            if try_merge(source, target):
                improved = True
        
        return colors

    out = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        if time.time() > deadline:
            pass
            
        edges = instance(*spec)
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        colors = solve_instance(edges, adj)
        out.append(colors)
        
    return out
