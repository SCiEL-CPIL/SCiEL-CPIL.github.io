import time

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin
    
    # Try algebraic construction for n=25 using F_5 x F_5
    def try_algebraic_n25():
        p = 5
        n = p * p
        colors = []
        # Precompute squares in F_p
        sq = [(x * x) % p for x in range(p)]
        
        for i in range(n):
            x1, y1 = divmod(i, p)
            for j in range(i + 1, n):
                x2, y2 = divmod(j, p)
                dx = (x2 - x1) % p
                dy = (y2 - y1) % p
                # Find color c such that dy = c * dx^2 mod p
                # If dx == 0, then dy must be 0 for any c, but dx=0 means x1=x2, dy=y2-y1
                # If dx=0: dy = c*0 = 0, so only works if dy=0, but then it's the same point
                # Actually if dx=0 and dy!=0, no c works via this formula
                # We need a different approach for vertical edges
                
                if dx == 0:
                    # Vertical edge: x1 == x2, dy != 0
                    # Assign based on dy
                    c = dy % p  # dy is in 1..4 since i!=j and dx=0 means dy!=0
                else:
                    inv_dx_sq = pow(dx * dx, p - 2, p)  # modular inverse of dx^2
                    c = (dy * inv_dx_sq) % p
                
                colors.append(c)
        
        # Verify no monochromatic C4
        # Build adjacency for each color
        adj = [set() for _ in range(p)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                adj[c].add((i, j))
                idx += 1
        
        # Check C4: for each color, for each pair of vertices, count common neighbors
        for c in range(p):
            # Build adjacency lists
            adj_list = [set() for _ in range(n)]
            for (i, j) in adj[c]:
                adj_list[i].add(j)
                adj_list[j].add(i)
            for i in range(n):
                for j in range(i + 1, n):
                    common = adj_list[i] & adj_list[j]
                    if len(common) >= 2:
                        return None  # C4 found
        return {'n': n, 'colors': colors}
    
    result = try_algebraic_n25()
    if result is not None and time.time() < deadline:
        return result
    
    # Fallback: the known n=11 construction
    n = 11
    return {'n': n, 'colors': [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]}
