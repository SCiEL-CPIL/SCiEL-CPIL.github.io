import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 5s buffer
    
    # Try to find a valid 5-coloring for n=16 first
    # If successful, try to extend to n=17, 18, etc.
    
    def try_construct(n):
        # edges in lexicographic order
        # We will use backtracking
        # State: colors array (list of length n*(n-1)//2), adj matrices
        # adj[c][i] is a bitmask of neighbors of i in color c
        
        max_n = n
        # Initialize
        adj = [[0] * n for _ in range(5)]
        colors = [0] * (n * (n - 1) // 2)
        edge_idx = 0
        
        # We need to map (i,j) to index
        # idx = i*n - (i+1)*i//2 + (j - i - 1) ? 
        # Simpler: just iterate i, j and keep a counter
        
        # Backtracking
        # To speed up, we can fix the first few edges
        # Edge (0,1) -> color 0
        
        def backtrack(i, j):
            nonlocal edge_idx
            if time.time() > deadline:
                return False
            
            if i == n - 1 and j == n:
                return True
            
            # Get current edge (i,j)
            # Calculate index
            # Number of edges before i: sum_{k=0}^{i-1} (n-1-k) = i*n - i*(i+1)//2
            # Edges in row i before j: j - i - 1
            idx = i * n - i * (i + 1) // 2 + (j - i - 1)
            
            found = False
            # Try colors 0 to 4
            # Symmetry breaking: if i==0 and j==1, only try 0
            # if i==0 and j==2, try 0,1
            # if i==0 and j==3, try 0,1,2
            # This is a standard trick for edge colorings
            
            for c in range(5):
                if i == 0 and j == 1 and c != 0: continue
                if i == 0 and j == 2 and c > 1: continue
                if i == 0 and j == 3 and c > 2: continue
                
                # Check constraint: common neighbors of i and j in color c < 2
                common = adj[c][i] & adj[c][j]
                # Bit count
                if common & (common - 1): # If more than 1 bit is set
                    continue
                
                # Assign
                colors[idx] = c
                adj[c][i] |= (1 << j)
                adj[c][j] |= (1 << i)
                
                # Next edge
                if j + 1 < n:
                    if backtrack(i, j + 1):
                        return True
                else:
                    if i + 1 < n - 1:
                        if backtrack(i + 1, i + 2):
                            return True
                
                # Undo
                adj[c][i] &= ~(1 << j)
                adj[c][j] &= ~(1 << i)
                colors[idx] = 0
                found = True # We tried at least one, but failed
            
            return False
        
        if backtrack(0, 1):
            return {'n': n, 'colors': colors}
        return None

    # Try n=16
    res = try_construct(16)
    if res:
        return res
        
    # If 16 failed (shouldn't), try 15
    res = try_construct(15)
    if res:
        return res
        
    # Fallback: Return a trivial valid coloring for n=1? Or n=2?
    # n=1 is always valid.
    return {'n': 1, 'colors': []}
