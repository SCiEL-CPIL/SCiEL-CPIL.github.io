    def is_valid_color_fast(colors, x, c):
        for idxs in ap_data[x]:
            c1 = colors[idxs[0]]
            c2 = colors[idxs[1]]
            c3 = colors[idxs[2]]
            
            # Fast distinct count check
            # Count distinct in c1, c2, c3
            if c1 == c2:
                if c2 == c3:
                    return False
                # c1 == c2 != c3
                if c == c1 or c == c3:
                    return False
            elif c1 == c3:
                # c1 == c3 != c2
                if c == c1 or c == c2:
                    return False
            elif c2 == c3:
                # c2 == c3 != c1
                if c == c1 or c == c2:
                    return False
            else:
                # c1, c2, c3 all distinct
                # Set size is 3. Adding c makes it 3 or 4. Always >= 3.
                pass
                
        return True
