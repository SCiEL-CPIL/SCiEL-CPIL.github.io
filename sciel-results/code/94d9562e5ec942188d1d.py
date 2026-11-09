def construct():
        edges = []
        # A = 0, 1, 2
        # B = 3..15
        # K3,13
        for a in range(3):
            for b in range(3, 16):
                edges.append([a, b])
        # Edges within A
        edges.append([0, 1])
        edges.append([1, 2])
        edges.append([0, 2])
        # Matching in B
        edges.append([3, 4])
        edges.append([5, 6])
        edges.append([7, 8])
        edges.append([9, 10])
        edges.append([11, 12])
        edges.append([13, 14])
        return edges
