        for y in S:
            old_dp_y = dp_removed[y]
            new_dp_y = [0] * (len(new_S)) # Size L+1
            
            # Copy old values
            # old_dp_y has size L.
            # new_dp_y has size L+1.
            for k in range(len(old_dp_y)):
                new_dp_y[k] = old_dp_y[k]
            
            # Add x
            # Max subset size of new_S \ {y} is L.
            # So we need to update up to index L.
            for k in range(len(new_S) - 1, 0, -1): # L down to 1
                 if new_dp_y[k-1]:
                     new_dp_y[k] |= (new_dp_y[k-1] << x)
            
            new_dp_removed[y] = new_dp_y
