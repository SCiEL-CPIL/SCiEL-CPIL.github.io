import numpy as np
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 290  # 10s safety margin
    
    # 1. Initialize Paley graph on 37 vertices
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    n = q
    # Adjacency matrix
    A = np.zeros((n, n), dtype=np.int8)
    for i in range(n):
        for j in range(i + 1, n):
            if (j - i) % q in residues:
                A[i, j] = 1
                A[j, i] = 1
                
    # Helper to get neighbors
    def get_neighbors(i):
        return np.where(A[i] == 1)[0]
        
    # Helper to check if a set of vertices forms a K4 or Independent 4
    # We will maintain lists of K4s and I4s for fast checking
    
    def find_k4s(A, n):
        k4s = []
        for i in range(n):
            for j in range(i+1, n):
                if A[i,j]:
                    for k in range(j+1, n):
                        if A[i,k] and A[j,k]:
                            for l in range(k+1, n):
                                if A[i,l] and A[j,l] and A[k,l]:
                                    k4s.append((i,j,k,l))
        return k4s
        
    def find_i4s(A, n):
        i4s = []
        for i in range(n):
            for j in range(i+1, n):
                if not A[i,j]:
                    for k in range(j+1, n):
                        if not A[i,k] and not A[j,k]:
                            for l in range(k+1, n):
                                if not A[i,l] and not A[j,l] and not A[k,l]:
                                    i4s.append((i,j,k,l))
        return i4s
        
    k4s = find_k4s(A, n)
    i4s = find_i4s(A, n)
    
    best_n = n
    
    # Try to add vertices
    while time.time() < deadline:
        # Try to add a new vertex
        added = False
        # Random restarts for finding a valid neighbor set S
        for _ in range(1000):
            # Randomly decide neighbors
            # Heuristic: try to balance degree
            S_mask = np.random.choice([0, 1], size=n, p=[0.5, 0.5])
            
            # Check if S contains any K4
            valid = True
            for k4 in k4s:
                if S_mask[k4[0]] and S_mask[k4[1]] and S_mask[k4[2]] and S_mask[k4[3]]:
                    valid = False
                    break
            if not valid:
                continue
                
            # Check if complement of S contains any I4
            T_mask = 1 - S_mask
            for i4 in i4s:
                if T_mask[i4[0]] and T_mask[i4[1]] and T_mask[i4[2]] and T_mask[i4[3]]:
                    valid = False
                    break
            if not valid:
                continue
                
            # If valid, add the vertex
            new_n = n + 1
            new_A = np.zeros((new_n, new_n), dtype=np.int8)
            new_A[:n, :n] = A
            new_A[:n, n] = S_mask
            new_A[n, :n] = S_mask
            
            A = new_A
            n = new_n
            best_n = n
            added = True
            
            # Update K4s and I4s
            # New K4s must include the new vertex n-1
            new_k4s = k4s.copy()
            new_i4s = i4s.copy()
            
            # Find new K4s involving n-1
            # A new K4 is {n-1, i, j, k} where i,j,k are in S and form a K3 in G[S]
            # Actually, we need i,j,k to be in S and form a triangle in the induced subgraph of S
            S_vertices = np.where(S_mask == 1)[0]
            for i, j, k in combinations(S_vertices, 3):
                if A[i,j] and A[j,k] and A[i,k]:
                    new_k4s.append((i, j, k, n-1))
                    
            # Find new I4s involving n-1
            # A new I4 is {n-1, i, j, k} where i,j,k are in T and form an independent set in G[T]
            T_vertices = np.where(S_mask == 0)[0]
            for i, j, k in combinations(T_vertices, 3):
                if not A[i,j] and not A[j,k] and not A[i,k]:
                    new_i4s.append((i, j, k, n-1))
                    
            k4s = new_k4s
            i4s = new_i4s
            break # Successfully added, restart loop to try adding another
            
        if not added:
            # If we can't add a vertex, try to modify the graph to make it easier
            # This is a simple heuristic: flip a random edge and see if it reduces constraints
            # For simplicity, we might just break if we can't add easily, 
            # but let's try a few random edge flips
            modified = False
            for _ in range(50):
                i, j = random.sample(range(n), 2)
                # Flip edge (i,j)
                A[i,j] = 1 - A[i,j]
                A[j,i] = A[i,j]
                
                # Check if graph is still valid (no K5, no I5)
                # We don't have K5 list, so we must check
                # Efficient check: 
                # If we flip an edge, we might create a K5 or I5.
                # We can check locally.
                # For now, let's just recompute K4s and I4s and check if any K5 or I5 exists
                # This is expensive. 
                # Alternative: Just try to add a vertex again. If it fails, revert.
                
                # Revert for now to keep code simple and safe
                A[i,j] = 1 - A[i,j]
                A[j,i] = A[i,j]
                
            break # If we can't add, stop to avoid wasting time
            
    # Convert to edge list
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if A[i,j] == 1:
                edges.append((i,j))
                
    return n, edges
