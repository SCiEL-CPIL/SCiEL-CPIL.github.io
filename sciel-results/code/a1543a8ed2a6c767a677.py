import random

def construct():
        n = 20
        best_edge_dict = {}
        
        # Run multiple attempts to find a dense valid coloring
        for _ in range(20):
            random.seed()
            candidates = [(u, v) for u in range(n) for v in range(u+1, n)]
            random.shuffle(candidates)
            
            edge_colors = {}
            
            for u, v in candidates:
                key = (min(u, v), max(u, v))
                valid_color = None
                for c in range(20):
                    ok = True
                    # Check for any 4-cycle u-a-b-v-u
                    # a, b must be distinct and not u, v
                    for a in range(n):
                        if a == u or a == v:
                            continue
                        key_ua = (min(u, a), max(u, a))
                        if key_ua not in edge_colors:
                            continue
                        c_ua = edge_colors[key_ua]
                        
                        for b in range(n):
                            if b == u or b == v or b == a:
                                continue
                            key_ab = (min(a, b), max(a, b))
                            if key_ab not in edge_colors:
                                continue
                            c_ab = edge_colors[key_ab]
                            
                            key_bv = (min(b, v), max(b, v))
                            if key_bv not in edge_colors:
                                continue
                            c_bv = edge_colors[key_bv]
                            
                            # Check distinctness
                            if c == c_ua or c == c_ab or c == c_bv:
                                ok = False
                                break
                            # Also c_ua, c_ab, c_bv must be distinct among themselves?
                            # Yes, the cycle must be rainbow.
                            # But wait, if c_ua == c_ab, the existing graph already had a non-rainbow 4-cycle?
                            # No, u-a-b-? is not a cycle yet.
                            # However, if we are adding u-v, and the path u-a-b-v exists,
                            # the cycle is u-a-b-v-u.
                            # The edges are ua, ab, bv, vu.
                            # We need all 4 distinct.
                            # We checked c (vu) against the others.
                            # Do we need to check ua, ab, bv against each other?
                            # If ua == ab, then the cycle has repeated colors.
                            # Is it possible that the existing graph has a path u-a-b-v where ua == ab?
                            # Yes.
                            # So we must check len({c_ua, c_ab, c_bv, c}) == 4.
                            
                            if len({c_ua, c_ab, c_bv, c}) < 4:
                                ok = False
                                break
                        if not ok:
                            break
                    if ok:
                        valid_color = c
                        break
                
                if valid_color is not None:
                    edge_colors[key] = valid_color
            
            if len(edge_colors) > len(best_edge_dict):
                best_edge_dict = edge_colors
        
        result = []
        for (u, v), c in best_edge_dict.items():
            result.append([u, v, c])
        
        return result
