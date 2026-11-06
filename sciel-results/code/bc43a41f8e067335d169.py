import time
import sys

N = 16

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # 50s limit for safety
    
    # Use a 2D list for simplicity, then flatten.
    color_matrix = [[0]*N for _ in range(N)]
    
    # Base case: K3 with 3 distinct colors
    # Vertices 0, 1, 2
    color_matrix[0][1] = 0
    color_matrix[1][0] = 0
    color_matrix[1][2] = 1
    color_matrix[2][1] = 1
    color_matrix[0][2] = 2
    color_matrix[2][0] = 2
    
    # Extend to K16
    # For each new vertex k from 3 to 15
    for k in range(3, N):
        # Color all edges (j, k) for j < k with color k
        for j in range(k):
            color_matrix[j][k] = k
            color_matrix[k][j] = k
            
    # Flatten matrix to list
    result = []
    for i in range(N):
        for j in range(i+1, N):
            result.append(color_matrix[i][j])
            
    return result

if __name__ == "__main__":
    # Just to test
    c = construct()
    print(len(set(c)))
