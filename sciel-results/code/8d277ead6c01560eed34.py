import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla
import math

def construct():
    N = 256
    # 1. Build the conflict graph
    # C[i][j] = 1 if i+1 and j+1 conflict
    # Conflict condition: (a*b) % (a+b) == 0
    # Equivalent to: let g=gcd(a,b), a=gx, b=gy, gcd(x,y)=1. Then (x+y) divides g.
    
    # We use 0-indexed for arrays, values 1..256
    conflict = np.zeros((N, N), dtype=np.uint8)
    
    for i in range(N):
        a = i + 1
        for j in range(i + 1, N):
            b = j + 1
            g = math.gcd(a, b)
            x = a // g
            y = b // g
            # Check if x+y divides g
            if g % (x + y) == 0:
                conflict[i, j] = 1
                conflict[j, i] = 1
                
    # 2. Build non-conflict adjacency matrix (for i != j)
    # A_nc[i][j] = 1 if not conflict
    # We want a large clique in A_nc, which is an independent set in conflict.
    # A_nc = J - I - conflict (for off-diagonal)
    # Since conflict is symmetric and 0 on diagonal:
    # A_nc[i][j] = 1 - conflict[i][j] for i != j
    # A_nc[i][i] = 0
    
    A_nc = 1 - conflict
    np.fill_diagonal(A_nc, 0)
    
    # 3. Spectral Analysis
    # Use scipy to find the principal eigenvector
    # A_nc is dense and small (256x256), so dense eigensolver is fine
    eigvals, eigvecs = np.linalg.eigh(A_nc)
    
    # The principal eigenvector corresponds to the largest eigenvalue
    idx = np.argmax(eigvals)
    v_main = eigvecs[:, idx]
    
    # We might also look at other eigenvectors if the main one is not discriminative enough,
    # but usually the largest eigenvalue's vector highlights the largest community/clique.
    
    # 4. Greedy Construction
    # Strategy: 
    # a) Sort vertices by eigenvector value (descending).
    # b) Add vertices in this order if they are compatible with the current set.
    
    order = np.argsort(v_main)[::-1]
    
    selected = []
    selected_set = set()
    
    for idx in order:
        vertex = idx
        # Check compatibility with all selected vertices
        compatible = True
        for u in selected:
            if conflict[vertex, u] == 1:
                compatible = False
                break
        if compatible:
            selected.append(vertex)
            selected_set.add(vertex)
            
    # 5. Local Search / Improvement
    # The greedy approach might miss some vertices.
    # Try to add any remaining vertices that fit.
    # Then try to swap: remove one vertex, add two? (2-exchange)
    # For N=256, a simple pass of trying to add all unselected vertices is good.
    
    # First, try to add any unselected vertex that fits
    for v in range(N):
        if v not in selected_set:
            compatible = True
            for u in selected:
                if conflict[v, u] == 1:
                    compatible = False
                    break
            if compatible:
                selected.append(v)
                selected_set.add(v)
                
    # Second, try simple 1-for-1 swaps (remove u, add v if v fits better with rest? 
    # Actually, just removing a vertex that allows adding 2 others might help, but 
    # given the structure found (136), the greedy spectral likely finds it.
    # Let's do a few iterations of:
    # - Try to add any unselected.
    # - If no add possible, try removing a vertex that has high degree in the 
    #   "conflict with unselected" sense, and see if we can add more.
    
    # Simple improvement loop
    improved = True
    while improved:
        improved = False
        # Try to add
        for v in range(N):
            if v not in selected_set:
                compatible = True
                for u in selected:
                    if conflict[v, u] == 1:
                        compatible = False
                        break
                if compatible:
                    selected.append(v)
                    selected_set.add(v)
                    improved = True
        
        # If no additions, try swaps
        if not improved:
            # For each vertex u in selected, temporarily remove it
            # and see if we can add more vertices than we lost.
            # This is O(N^3) worst case, but N=256 is small.
            # We limit this to a few iterations to avoid timeout.
            for u_idx in range(len(selected)):
                u = selected[u_idx]
                # Remove u
                temp_selected = selected[:u_idx] + selected[u_idx+1:]
                temp_set = set(temp_selected)
                
                added = 0
                new_selected = list(temp_selected)
                for v in range(N):
                    if v not in temp_set:
                        compatible = True
                        for w in new_selected:
                            if conflict[v, w] == 1:
                                compatible = False
                                break
                        if compatible:
                            new_selected.append(v)
                            temp_set.add(v)
                            added += 1
                
                if added > 1: # Net gain
                    selected = new_selected
                    selected_set = temp_set
                    improved = True
                    break # Restart loop with new state
                    
    # Convert to list of values 1..256
    result = [v + 1 for v in selected]
    
    # Ensure non-empty
    if not result:
        return [1]
        
    return result

