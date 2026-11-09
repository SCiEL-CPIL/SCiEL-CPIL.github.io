import math
from itertools import product

def construct():
    # We want to maximize sum(1/m) for m in [20, 200]
    # Strategy: Use multiples of 10: m = 10*k for k in [2, 20]
    # Condition: r_i != r_j mod 10*gcd(k_i, k_j)
    # If we ensure r_i != r_j mod 10 for all pairs with gcd(k_i, k_j)=1,
    # it implies r_i != r_j mod 10*g for all pairs (since if a!=b mod 10, a!=b mod 10g)
    
    ks = list(range(2, 21)) # 2 to 20
    n = len(ks)
    
    # Check if we can assign residues 0..9 to k's such that coprime pairs have different residues
    # This is a graph coloring problem with 10 colors
    
    # Build adjacency list for coprime pairs
    adj = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            if math.gcd(ks[i], ks[j]) == 1:
                adj[i].append(j)
                adj[j].append(i)
                
    # Backtracking search for valid coloring
    colors = [-1] * n
    
    def dfs(idx):
        if idx == n:
            return True
        # Try colors 0..9
        for c in range(10):
            # Check if color c is valid for idx
            valid = True
            for nb in adj[idx]:
                if colors[nb] == c:
                    valid = False
                    break
            if valid:
                colors[idx] = c
                if dfs(idx + 1):
                    return True
                colors[idx] = -1
        return False
        
    if dfs(0):
        # Found valid assignment
        classes = []
        for i in range(n):
            m = 10 * ks[i]
            r = colors[i]
            classes.append([m, r])
        return classes
    else:
        # Fallback: Use multiples of 20 (seed strategy)
        # This is guaranteed valid
        return [[20 * j, j - 1] for j in range(1, 11)]
