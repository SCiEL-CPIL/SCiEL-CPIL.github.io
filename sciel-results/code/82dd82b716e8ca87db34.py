import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    n_base = 29
    is_residue = [False] * n_base
    for x in range(1, n_base):
        is_residue[(x * x) % n_base] = True
    
    # colors_base[i][j] for i < j
    colors_base = [[0] * n_base for _ in range(n_base)]
    for i in range(n_base):
        for j in range(i + 1, n_base):
            diff = (j - i) % n_base
            if is_residue[diff]:
                colors_base[i][j] = 0
                colors_base[j][i] = 0
            else:
                colors_base[i][j] = 1
                colors_base[j][i] = 1

    n_target = 30
    
    # Precompute colors for edges in 0..28 for faster access in the inner loop
    # c[u][v]
    
    def count_violations_29(new_edges):
        count = 0
        # Iterate over all 5-subsets of 0..28
        # To speed up, we can pre-compute the list of combinations? No, just iterate.
        # We can also pre-compute the "rem" partitions? No, depends on k.
        
        # Optimization: 
        # For a fixed vk, the condition "all(new_edges[r] == c for r in rem)" 
        # means that new_edges must be constant on the set {vk} U rem = all 5 vertices.
        # So for a fixed combo of 5 vertices, if new_edges is not constant on these 5 vertices, 
        # we can skip all 5 choices of vk?
        # No, vk is part of the 5.
        # If new_edges is not constant on the 5 vertices, then for any vk, the set rem 
        # (which is the other 4) might not be constant with vk.
        # Actually, the condition is:
        # new_edges[vk] == c AND all(new_edges[r] == c for r in rem).
        # This is equivalent to: all(new_edges[r] == c for r in combo).
        # So if the 5 vertices in combo do not have the same color in new_edges, 
        # then NO vk will work for that combo!
        
        # This is a huge optimization.
        # For each combo of 5 vertices:
        # Check if all new_edges[r] for r in combo are 0.
        # If yes, check the 3 partitions for color 0.
        # Check if all new_edges[r] for r in combo are 1.
        # If yes, check the 3 partitions for color 1.
        # If mixed, skip.
        
        # This reduces the inner work significantly.
        
        for combo in itertools.combinations(range(29), 5):
            # Check if all 0
            all_0 = True
            all_1 = True
            for r in combo:
                if new_edges[r] != 0: all_0 = False
                if new_edges[r] != 1: all_1 = False
            
            if all_0:
                # Check 3 partitions for color 0
                # rem is just the list of 5, but we don't need to pick vk anymore?
                # Wait, the partition structure depends on which vertex is vk.
                # But the condition "all edges between pairs are 0" depends on the specific pairs.
                # The pairs are formed by partitioning the 5 vertices into {vk} and {r1, r2, r3, r4}
                # and then partitioning {r1..r4} into two pairs.
                # So we still need to iterate over vk (5 choices) and partition of rem (3 choices).
                # But the edge checks are the same.
                
                # We can just iterate vk and partitions.
                for k in range(5):
                    vk = combo[k]
                    rem = [combo[i] for i in range(5) if i != k]
                    
                    # Part 1
                    p1, p2 = (rem[0], rem[1]), (rem[2], rem[3])
                    if (colors_base[p1[0]][p1[1]] == 0 and colors_base[p2[0]][p2[1]] == 0 and
                        colors_base[p1[0]][p2[0]] == 0 and colors_base[p1[0]][p2[1]] == 0 and
                        colors_base[p1[1]][p2[0]] == 0 and colors_base[p1[1]][p2[1]] == 0):
                        count += 1
                    # Part 2
                    p1, p2 = (rem[0], rem[2]), (rem[1], rem[3])
                    if (colors_base[p1[0]][p1[1]] == 0 and colors_base[p2[0]][p2[1]] == 0 and
                        colors_base[p1[0]][p2[0]] == 0 and colors_base[p1[0]][p2[1]] == 0 and
                        colors_base[p1[1]][p2[0]] == 0 and colors_base[p1[1]][p2[1]] == 0):
                        count += 1
                    # Part 3
                    p1, p2 = (rem[0], rem[3]), (rem[1], rem[2])
                    if (colors_base[p1[0]][p1[1]] == 0 and colors_base[p2[0]][p2[1]] == 0 and
                        colors_base[p1[0]][p2[0]] == 0 and colors_base[p1[0]][p2[1]] == 0 and
                        colors_base[p1[1]][p2[0]] == 0 and colors_base[p1[1]][p2[1]] == 0):
                        count += 1
            
            if all_1:
                for k in range(5):
                    vk = combo[k]
                    rem = [combo[i] for i in range(5) if i != k]
                    
                    # Part 1
                    p1, p2 = (rem[0], rem[1]), (rem[2], rem[3])
                    if (colors_base[p1[0]][p1[1]] == 1 and colors_base[p2[0]][p2[1]] == 1 and
                        colors_base[p1[0]][p2[0]] == 1 and colors_base[p1[0]][p2[1]] == 1 and
                        colors_base[p1[1]][p2[0]] == 1 and colors_base[p1[1]][p2[1]] == 1):
                        count += 1
                    # Part 2
                    p1, p2 = (rem[0], rem[2]), (rem[1], rem[3])
                    if (colors_base[p1[0]][p1[1]] == 1 and colors_base[p2[0]][p2[1]] == 1 and
                        colors_base[p1[0]][p2[0]] == 1 and colors_base[p1[0]][p2[1]] == 1 and
                        colors_base[p1[1]][p2[0]] == 1 and colors_base[p1[1]][p2[1]] == 1):
                        count += 1
                    # Part 3
                    p1, p2 = (rem[0], rem[3]), (rem[1], rem[2])
                    if (colors_base[p1[0]][p1[1]] == 1 and colors_base[p2[0]][p2[1]] == 1 and
                        colors_base[p1[0]][p2[0]] == 1 and colors_base[p1[0]][p2[1]] == 1 and
                        colors_base[p1[1]][p2[0]] == 1 and colors_base[p1[1]][p2[1]] == 1):
                        count += 1
                        
        return count

    # Local search
    best_new_edges = [random.randint(0, 1) for _ in range(29)]
    best_count = count_violations_29(best_new_edges)
    
    while time.time() < deadline:
        idx = random.randint(0, 28)
        new_edges = best_new_edges.copy()
        new_edges[idx] = 1 - new_edges[idx]
        
        new_count = count_violations_29(new_edges)
        
        if new_count < best_count:
            best_new_edges = new_edges
            best_count = new_count
            if best_count == 0:
                break
    
    if best_count == 0:
        colors = []
        for i in range(29):
            for j in range(i + 1, 29):
                colors.append(colors_base[i][j])
        for i in range(29):
            colors.append(best_new_edges[i])
        return {'n': 30, 'colors': colors}
    else:
        colors = []
        for i in range(29):
            for j in range(i + 1, 29):
                colors.append(colors_base[i][j])
        return {'n': 29, 'colors': colors}
