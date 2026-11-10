    def _is_2_colorable(n, edges):
        if not edges: return True
        colors = [-1] * n
        # Precompute edge indices for each vertex
        v_to_e = [[] for _ in range(n)]
        for i, e in enumerate(edges):
            for v in e:
                v_to_e[v].append(i)
                
        # State: count of 0s and 1s in each edge
        c0 = [0] * len(edges)
        c1 = [0] * len(edges)
        
        def propagate(start_v, start_c):
            # Queue of (vertex, color) to process
            # Or just process edges affected by changed vertices
            # We need a queue of edges to check
            q = []
            visited_e = set()
            
            # Initial setup is handled by caller setting colors[start_v] = start_c
            # We need to update counts for edges containing start_v
            for ei in v_to_e[start_v]:
                if colors[start_v] == 0: c0[ei] += 1
                else: c1[ei] += 1
                q.append(ei)
                visited_e.add(ei)
                
            while q:
                ei = q.pop()
                # Check edge ei
                if c0[ei] == 5: return False
                if c1[ei] == 5: return False
                
                if c0[ei] == 4:
                    # Find the uncolored vertex
                    for v in edges[ei]:
                        if colors[v] == -1:
                            # Must be 1
                            colors[v] = 1
                            c1[ei] += 1 # Update current edge? No, c1[ei] becomes 5?
                            # Wait, if c0==4 and we set last to 1, c1 becomes 1?
                            # No, c0=4, c1=0, uncolored=1. Set to 1 -> c1=1.
                            # But we need to propagate to other edges.
                            # Update counts for other edges containing v
                            for nei in v_to_e[v]:
                                if nei != ei:
                                    c1[nei] += 1
                                    if nei not in visited_e:
                                        q.append(nei)
                                        visited_e.add(nei)
                            # Also check if this edge now has c1=5? No, c1 was 0, now 1.
                            # But what if c1 was already 4? Then c0 must be 0?
                            # If c0=4 and c1=4 is impossible for 5 vertices.
                            # So if c0=4, c1 must be 0 or 1?
                            # If c0=4, the 5th is uncolored.
                            # If we set it to 1, c1 becomes 1.
                            # Is it possible that c1 was already 1? No, because 4+1=5 vertices.
                            # So c1 must have been 0.
                            # So no contradiction here.
                            pass
                        elif colors[v] == 1:
                            # Contradiction?
                            # If c0=4, and one vertex is 1, then c1 >= 1.
                            # Total colored = c0 + c1 = 5. So all colored.
                            # But we found an uncolored vertex in the loop?
                            # No, the loop finds the uncolored one.
                            # If we find a vertex with color 1, it means c1 >= 1.
                            # If c0=4 and c1>=1, sum >= 5.
                            # If sum == 5, all are colored.
                            # If sum > 5, contradiction.
                            # But we are iterating to find the uncolored one.
                            # If we find a colored one, we don't care unless it's a contradiction.
                            # Actually, if c0=4, there is exactly 1 uncolored.
                            # So the loop will find exactly 1 uncolored and 4 colored.
                            # If the 4 colored are all 0, then c1=0.
                            # If any are 1, c1>0.
                            # If c1>0 and c0=4, then c0+c1 > 4.
                            # If c0+c1 == 5, all colored. But we are looking for uncolored.
                            # So if c0=4, and we find a vertex with color 1, it must be that c1>=1.
                            # Then c0+c1 >= 5.
                            # If c0+c1 == 5, no uncolored.
                            # But we are in the branch `if c0[ei] == 4`.
                            # This implies there is 1 uncolored.
                            # So the 4 colored must be 0s.
                            # So we should not encounter a 1.
                            # If we do, it's a logic error in state tracking.
                            pass
                elif c1[ei] == 4:
                    # Symmetric
                    pass
            return True
            
        def backtrack():
            # Find uncolored vertex
            # Heuristic: Vertex in most "tight" edges?
            # Just first uncolored.
            v = -1
            for i in range(n):
                if colors[i] == -1:
                    v = i
                    break
            if v == -1:
                return True
                
            # Try 0
            colors[v] = 0
            # Propagate
            # We need a robust propagate that handles the queue correctly.
            # The above sketch is a bit messy. Let's use a cleaner implementation.
            pass
    