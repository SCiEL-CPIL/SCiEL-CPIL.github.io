import time
import random
from itertools import combinations

def construct():
    deadline = time.time() + 140
    n = 16
    edge_idx = [(i, j) for i in range(n) for j in range(i + 1, n)]
    num_edges = len(edge_idx)
    edge_to_idx = {e: idx for idx, e in enumerate(edge_idx)}
    
    k4s = list(combinations(range(n), 4))
    k4_edges = []
    for k4 in k4s:
        edges = []
        for a, b in combinations(k4, 2):
            edges.append(edge_to_idx[(a, b)])
        k4_edges.append(edges)
        
    edge_to_k4s = [[] for _ in range(num_edges)]
    for k4_idx, edges in enumerate(k4_edges):
        for e in edges:
            edge_to_k4s[e].append(k4_idx)

    # State: colors, bad_k4s (set of indices of K4s with < 5 colors)
    # We need a function to compute colors in a K4
    def get_k4_colors(colors, edges):
        return [colors[e] for e in edges]

    def is_bad(colors, edges):
        return len(set(colors[e] for e in edges)) < 5

    # Initialize
    colors = [random.randint(0, 16) for _ in range(num_edges)]
    
    # Compute initial bad K4s
    bad_k4s = set()
    for i, edges in enumerate(k4_edges):
        if is_bad(colors, edges):
            bad_k4s.add(i)
            
    # Local Search to fix violations
    max_iterations = 50000
    for _ in range(max_iterations):
        if time.time() > deadline:
            break
        if not bad_k4s:
            break
            
        # Pick a random bad K4
        bad_idx = random.choice(list(bad_k4s))
        edges = k4_edges[bad_idx]
        
        # Try to change one edge in this K4
        # Pick a random edge from the K4
        e = random.choice(edges)
        old_c = colors[e]
        
        # Try a new color
        # To fix a K4 with < 5 colors, we want to introduce a new color or change a duplicate
        # Random is fine
        new_c = random.randint(0, 16)
        if new_c == old_c:
            continue
            
        colors[e] = new_c
        
        # Update bad_k4s for all K4s containing e
        # First, remove e's K4s from consideration? No, just re-evaluate them.
        # For each K4 containing e:
        # 1. If it was bad, check if it is now good. If good, remove from bad_k4s.
        # 2. If it was good, check if it is now bad. If bad, add to bad_k4s.
        
        for k4_idx in edge_to_k4s[e]:
            if k4_idx == bad_idx:
                continue # We are handling this one, but it's safer to include
            edges_k4 = k4_edges[k4_idx]
            was_bad = k4_idx in bad_k4s
            is_now_bad = is_bad(colors, edges_k4)
            
            if was_bad and not is_now_bad:
                bad_k4s.remove(k4_idx)
            elif not was_bad and is_now_bad:
                bad_k4s.add(k4_idx)
                
        # Handle the specific bad_idx we picked
        # It might have become good
        if bad_idx in bad_k4s:
            if not is_bad(colors, edges):
                bad_k4s.remove(bad_idx)
            else:
                # Still bad, revert? Or keep and try another?
                # Keeping it might help if we change another edge later.
                # But if we don't make progress, we might loop.
                # Let's revert if it's still bad to avoid oscillation?
                # Or just leave it.
                pass

    # If we found a valid solution, try to minimize colors
    if not bad_k4s:
        # Greedy Merging
        current_colors = colors[:]
        current_used = set(current_colors)
        
        while len(current_used) > 13:
            if time.time() > deadline:
                break
            freq = {}
            for c in current_colors:
                freq[c] = freq.get(c, 0) + 1
            least_used = min(freq, key=freq.get)
            
            merged = False
            targets = [c for c in current_used if c != least_used]
            random.shuffle(targets)
            
            for t in targets:
                new_colors = [t if c == least_used else c for c in current_colors]
                # Check validity
                # This is O(1820 * 6)
                valid = True
                for edges in k4_edges:
                    if len(set(new_colors[e] for e in edges)) < 5:
                        valid = False
                        break
                if valid:
                    current_colors = new_colors
                    current_used = set(current_colors)
                    merged = True
                    break
                    
            if not merged:
                break
                
        colors = current_colors

    return colors
