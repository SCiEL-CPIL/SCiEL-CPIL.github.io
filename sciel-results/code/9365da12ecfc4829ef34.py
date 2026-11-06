import itertools
import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n = 16
    vertices = list(range(n))
    subsets_4 = list(itertools.combinations(vertices, 4))
    num_subsets = len(subsets_4)
    
    edges = [(i,j) for i in range(n) for j in range(i+1, n)]
    num_edges = len(edges)
    edge_index = {e: idx for idx, e in enumerate(edges)}
    
    subset_edges = []
    edge_to_subsets = [[] for _ in range(num_edges)]
    
    for s_idx, s in enumerate(subsets_4):
        a,b,c,d = s
        e_indices = [
            edge_index[(a,b)], edge_index[(a,c)], edge_index[(a,d)],
            edge_index[(b,c)], edge_index[(b,d)], edge_index[(c,d)]
        ]
        subset_edges.append(e_indices)
        for e_idx in e_indices:
            edge_to_subsets[e_idx].append(s_idx)
            
    def count_colors(colours, e_indices):
        c1 = colours[e_indices[0]]
        c2 = colours[e_indices[1]]
        c3 = colours[e_indices[2]]
        c4 = colours[e_indices[3]]
        c5 = colours[e_indices[4]]
        c6 = colours[e_indices[5]]
        return len({c1,c2,c3,c4,c5,c6})

    def is_valid(colours):
        for s_idx in range(num_subsets):
            if count_colors(colours, subset_edges[s_idx]) < 5:
                return False
        return True

    # Finite geometry approach: map vertices to F_4^2
    # F_4 = {0, 1, alpha, alpha+1} where alpha^2 = alpha + 1
    # Elements: 0->0, 1->1, 2->alpha, 3->alpha+1
    # Addition table in F_4:
    #   0+0=0, 0+1=1, 0+2=2, 0+3=3
    #   1+1=0, 1+2=3, 1+3=2
    #   2+2=0, 2+3=1
    #   3+3=0
    # Multiplication table in F_4:
    #   0*x=0 for all x
    #   1*x=x for all x
    #   2*2=3, 2*3=1, 2*1=2
    #   3*3=2, 3*1=3, 3*2=1
    
    f4_add = [
        [0,1,2,3],
        [1,0,3,2],
        [2,3,0,1],
        [3,2,1,0]
    ]
    f4_mul = [
        [0,0,0,0],
        [0,1,2,3],
        [0,2,3,1],
        [0,3,1,2]
    ]
    
    def f4_inv(x):
        if x == 0:
            return None
        return 4 - x if x in (1,3) else (3 if x == 2 else 1)
        # Actually: 1^-1=1, 2^-1=2, 3^-1=3 in F_4? Let me verify:
        # 2*2=3, 2*3=1, so 2^-1=3
        # 3*3=2, 3*2=1, so 3^-1=2
        # 1*1=1, so 1^-1=1
    
    def f4_inv_correct(x):
        if x == 0:
            return None
        if x == 1:
            return 1
        if x == 2:
            return 3
        if x == 3:
            return 2
    
    # Map 16 vertices to F_4^2
    vertex_points = [(i//4, i%4) for i in range(16)]
    
    # Color edges by slope in F_4
    # For points p1=(x1,y1), p2=(x2,y2):
    # If x1 != x2: slope = (y2-y1)/(x2-x1) in F_4
    # If x1 == x2: slope = infinity (vertical)
    # There are 4 finite slopes + 1 infinite = 5 slope classes
    
    # But we want more colors. Let's use directed slopes or add another feature.
    # Actually, let's first try just slopes and see how many colors that gives.
    # 5 colors might not be enough. Let's think...
    
    # Alternative: color by the pair (slope, something_else)
    # Or: use a different structure.
    
    # Let me try: color edge (i,j) by the slope of the line through them in F_4^2.
    # This gives 5 colors. But we need at least 13. So this alone won't work.
    
    # Better idea: Use the fact that in F_4^2, we can define multiple affine structures.
    # Or: color by (slope, intercept) which gives more colors.
    
    # Actually, let's try a different approach:
    # Map vertices to F_4^2 and color edge (i,j) by the value of a quadratic form or bilinear form.
    
    # Let's try: color(i,j) = f(point_i, point_j) for some function f.
    
    # Simplest: color by the difference vector (dx, dy) encoded as a single number.
    # dx in F_4 (4 values), dy in F_4 (4 values), but we need to handle the case where dx=dy=0 (same point).
    # For distinct points, (dx,dy) != (0,0). There are 15 possible nonzero vectors.
    # But (dx,dy) and (-dx,-dy) give the same edge. So we have 15/... hmm, this is getting complicated.
    
    # Let's just try the slope approach first and see if it gives a valid coloring with few colors.
    
    def slope_color(p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        dx = f4_add[x2, 4-x1]  # x2 - x1
        dy = f4_add[y2, 4-y1]  # y2 - y1
        # Wait, subtraction in F_4: a - b = a + (-b). In F_4, -0=0, -1=3, -2=2, -3=1
        # So -x = 4-x for x in {1,2,3}, and -0=0
        # Let me redo this properly.
        
    f4_neg = [0, 3, 2, 1]  # negation: -0=0, -1=3, -2=2, -3=1
    
    def f4_sub(a, b):
        return f4_add[a, f4_neg[b]]
    
    def slope_of(p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        dx = f4_sub(x2, x1)
        dy = f4_sub(y2, y1)
        if dx == 0:
            return 4  # infinity
        inv_dx = f4_inv_correct(dx)
        slope = f4_mul[dy, inv_dx]
        return slope  # 0, 1, 2, 3
    
    # This gives 5 colors (0,1,2,3,4). Too few.
    
    # Let's try a different coloring: use the full difference vector.
    # Encode (dx, dy) where (dx,dy) != (0,0) as a single number 0..14.
    # But we need to identify (dx,dy) with (-dx,-dy) since they represent the same edge.
    # Actually no, the edge is undirected, so (i,j) and (j,i) are the same.
    # The difference (dx,dy) for (i,j) is the negation of the difference for (j,i).
    # So we need a canonical representative.
    
    # Simpler: just use all 15 nonzero vectors as colors, but identify v and -v.
    # That gives 15/2 = 7.5, which doesn't work. So we can't do that.
    
    # Alternative: don't identify v and -v. Just use the vector from the smaller-index vertex to the larger-index vertex.
    # This gives 15 colors. Let's try that.
    
    def vec_color(i, j):
        p1 = vertex_points[i]
        p2 = vertex_points[j]
        x1, y1 = p1
        x2, y2 = p2
        dx = f4_sub(x2, x1)
        dy = f4_sub(y2, y1)
        # Encode (dx, dy) as dx*4 + dy, but ensure it's nonzero
        return dx * 4 + dy
    
    colours = [vec_color(i, j) for i in range(n) for j in range(i+1, n)]
    # This gives colors in range 0..15, but (0,0) is excluded since i != j.
    # Actually, dx=0,dy=0 would mean same point, which can't happen.
    # The colors used will be a subset of {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}
    # Wait, dx*4+dy where (dx,dy) != (0,0):
    # dx=0: dy=1,2,3 -> colors 1,2,3
    # dx=1: dy=0,1,2,3 -> colors 4,5,6,7
    # dx=2: dy=0,1,2,3 -> colors 8,9,10,11
    # dx=3: dy=0,1,2,3 -> colors 12,13,14,15
    # So 15 colors total.
    
    if is_valid(colours):
        # Count distinct colors used
        distinct = len(set(colours))
        if distinct <= 15:
            return colours
    
    # Try slope-based coloring with 5 colors, combined with something else
    # Or just try local search with the vector coloring as initial
    
    def local_search(m, max_time=10.0, initial=None):
        start_ls = time.time()
        best_viol = float('inf')
        best_colours = None
        
        for restart in range(100):
            if time.time() - start_ls > max_time:
                break
            if time.time() > deadline:
                break
                
            if restart == 0 and initial is not None:
                colours = initial[:]
            else:
                colours = [random.randint(0, m-1) for _ in range(num_edges)]
            
            # Calculate initial violations
            viol = [0] * num_subsets
            total_viol = 0
            for s_idx in range(num_subsets):
                e_idx_list = subset_edges[s_idx]
                if count_colors(colours, e_idx_list) < 5:
                    viol[s_idx] = 1
                    total_viol += 1
                    
            if total_viol == 0:
                return colours
                
            if total_viol < best_viol:
                best_viol = total_viol
                best_colours = colours[:]
                
            improved = True
            stall_counter = 0
            while improved and time.time() - start_ls < max_time:
                improved = False
                stall_counter += 1
                if stall_counter > 5000:
                    break
                    
                idx = random.randint(0, num_edges-1)
                old_c = colours[idx]
                
                affected_subsets = edge_to_subsets[idx]
                
                min_new_total = total_viol
                best_new_c = old_c
                
                for new_c in range(m):
                    if new_c == old_c:
                        continue
                    
                    colours[idx] = new_c
                    
                    new_total = total_viol
                    for s_idx in affected_subsets:
                        was_viol = viol[s_idx]
                        e_idx_list = subset_edges[s_idx]
                        is_viol = 1 if count_colors(colours, e_idx_list) < 5 else 0
                        new_total += (is_viol - was_viol)
                        
                    if new_total < min_new_total:
                        min_new_total = new_total
                        best_new_c = new_c
                        
                colours[idx] = old_c
                
                if min_new_total < total_viol:
                    colours[idx] = best_new_c
                    for s_idx in affected_subsets:
                        was_viol = viol[s_idx]
                        e_idx_list = subset_edges[s_idx]
                        is_viol = 1 if count_colors(colours, e_idx_list) < 5 else 0
                        viol[s_idx] = is_viol
                        total_viol += (is_viol - was_viol)
                    
                    improved = True
                    if total_viol < best_viol:
                        best_viol = total_viol
                        best_colours = colours[:]
                    if total_viol == 0:
                        return colours
                else:
                    break

        return best_colours if best_viol == 0 else None
    
    # Try local search with vector coloring as initial for m=15
    res15 = local_search(15, max_time=30.0, initial=colours)
    if res15 is not None:
        return res15
    
    # Try m=14
    res14 = local_search(14, max_time=30.0)
    if res14 is not None:
        return res14
    
    # Try m=13
    res13 = local_search(13, max_time=30.0)
    if res13 is not None:
        return res13
    
    # Fallback
    return [(i + j) % 17 for i in range(n) for j in range(i + 1, n)]
