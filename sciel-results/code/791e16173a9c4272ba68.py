import time
import sys

N = 16

def construct():
    # 1-factorization of K_16
    # Vertices 0..14 and 15 (infinity)
    # Colors 0..14
    
    # color_matrix[i][j] will hold the color for edge (i,j)
    color_matrix = [[-1 for _ in range(N)] for _ in range(N)]
    
    # There are N-1 = 15 perfect matchings
    for k in range(N - 1):
        # The edge connecting infinity (15) to k
        color_matrix[15][k] = k
        color_matrix[k][15] = k
        
        # The other 7 edges in this matching
        # Pairs (k+j, k-j) mod 15 for j=1..7
        for j in range(1, (N - 2) // 2 + 1):
            u = (k + j) % (N - 1)
            v = (k - j) % (N - 1)
            color_matrix[u][v] = k
            color_matrix[v][u] = k
            
    # Build the result list in lexicographic order of pairs (i, j) with 0 <= i < j < 16
    result = []
    for i in range(N):
        for j in range(i + 1, N):
            result.append(color_matrix[i][j])
            
    return result
