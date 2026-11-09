import random
import time

def _local_search_27():
    n = 27
    # Start with random coloring
    colors = [random.randint(0, 4) for _ in range(n*(n-1)//2)]
    
    def count_c4s(colors):
        # Count total C4s across all colors
        total = 0
        idx = {}
        for i in range(n):
            for j in range(i+1, n):
                idx[(i,j)] = colors[i*(n-1) - i*(i+1)//2 + (j-i-1)]
        
        for c in range(5):
            adj = [set() for _ in range(n)]
            for (i,j), col in idx.items():
                if col == c:
                    adj[i].add(j)
                    adj[j].add(i)
            for i in range(n):
                si = adj[i]
                for j in si:
                    inter = si & adj[j]
                    inter.discard(i)
                    inter.discard(j)
                    total += len(inter)
        return total
    
    def fix_c4s(colors):
        # Find a C4 and fix it
        idx = {}
        for i in range(n):
            for j in range(i+1, n):
                idx[(i,j)] = colors[i*(n-1) - i*(i+1)//2 + (j-i-1)]
        
        for c in range(5):
            adj = [set() for _ in range(n)]
            for (i,j), col in idx.items():
                if col == c:
                    adj[i].add(j)
                    adj[j].add(i)
            for i in range(n):
                si = adj[i]
                for j in si:
                    inter = si & adj[j]
                    inter.discard(i)
                    inter.discard(j)
                    if inter:
                        k = next(iter(inter))
                        # C4: i-j-k-i? No, i and j share neighbor k.
                        # C4: i-k-j-? We need a fourth vertex.
                        # Actually, i and j have common neighbor k, so i-k-j is a path of length 2.
                        # A C4 would be i-a-j-b-i where a,b are common neighbors... 
                        # Wait, if i and j have two common neighbors k and l, then i-k-j-l-i is a C4.
                        # So we need to find two vertices with two common neighbors.
                        # Let me redo: for each pair (i,j), count common neighbors. If >=2, we have a C4.
                        # To fix, change one of the edges.
                        # Change edge (i,k) to a different color.
                        edge_idx = i*(n-1) - i*(i+1)//2 + (k-i-1) if i < k else k*(n-1) - k*(k+1)//2 + (i-k-1)
                        old_color = colors[edge_idx]
                        new_color = (old_color + 1) % 5
                        colors[edge_idx] = new_color
                        return True
        return False
    
    best = count_c4s(colors)
    for _ in range(10000):
        if best == 0:
            break
        if _time_left() < 5:
            break
        if fix_c4s(colors):
            new_count = count_c4s(colors)
            if new_count < best:
                best = new_count
            else:
                # Revert
                pass  # This is getting complicated
    return {'n': n, 'colors': colors}
