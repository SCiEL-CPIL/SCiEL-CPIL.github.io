import time
import sys

N = 16

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # 50s limit for safety
    
    # Start with base Gallai coloring: color(i,j) = min(i,j) for i != j
    color_matrix = [[0]*N for _ in range(N)]
    for i in range(N):
        for j in range(i+1, N):
            c = min(i, j)
            color_matrix[i][j] = c
            color_matrix[j][i] = c
    
    # Helper: check if recoloring edge (i,j) to new_color maintains validity
    # Only check 4-cycles that include both i and j
    def check_edge_recolor(cm, i, j, new_color):
        # Find all other pairs (k,l) from remaining vertices
        others = [v for v in range(N) if v != i and v != j]
        for idx_k in range(len(others)):
            for idx_l in range(idx_k + 1, len(others)):
                k = others[idx_k]
                l = others[idx_l]
                # Three 4-cycles on {i, j, k, l}
                # We need to check all 3 distinct 4-cycles
                # Cycle 1: i-k-j-l-i
                c1 = [cm[i][k], cm[k][j], cm[j][l], cm[l][i]]
                if len(set(c1)) == 4:
                    return False
                # Cycle 2: i-k-l-j-i
                c2 = [cm[i][k], cm[k][l], cm[l][j], cm[j][i]]
                if len(set(c2)) == 4:
                    return False
                # Cycle 3: i-l-k-j-i
                c3 = [cm[i][l], cm[l][k], cm[k][j], cm[j][i]]
                if len(set(c3)) == 4:
                    return False
        return True
    
    # Current number of colors
    current_colors = set()
    for i in range(N):
        for j in range(i+1, N):
            current_colors.add(color_matrix[i][j])
    next_color = max(current_colors) + 1 if current_colors else 0
    
    # Greedy: try to recolor edges with new colors
    # Try multiple passes to allow more colors
    edge_list = [(i, j) for i in range(N) for j in range(i+1, N)]
    
    # Try several passes
    for pass_num in range(10):
        if time.time() > deadline - 5:
            break
        for (i, j) in edge_list:
            if time.time() > deadline - 5:
                break
            
            old_color = color_matrix[i][j]
            
            # Try to assign a new color
            new_color = next_color
            color_matrix[i][j] = new_color
            color_matrix[j][i] = new_color
            
            if check_edge_recolor(color_matrix, i, j, new_color):
                # Keep the new color
                next_color += 1
            else:
                # Revert
                color_matrix[i][j] = old_color
                color_matrix[j][i] = old_color
    
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
