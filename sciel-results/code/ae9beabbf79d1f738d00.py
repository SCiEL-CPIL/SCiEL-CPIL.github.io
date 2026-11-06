        def construct():
            # Grötzsch graph edges
            edges = []
            # Cycle 0-4
            for i in range(5):
                edges.append([i, (i+1)%5])
            # Spokes
            for i in range(5):
                edges.append([i, 5+i])
            # Top
            for i in range(5):
                edges.append([10, 5+i])
            # Copy cycle
            for i in range(5):
                edges.append([5+i, 5+(i+1)%5])
            
            # Independent set
            indep = [0, 2, 4, 6, 8]
            
            # Add vertices 11-15
            for v in range(11, 16):
                for u in indep:
                    edges.append([v, u])
                    
            return edges
        