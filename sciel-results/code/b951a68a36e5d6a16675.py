        edges = []
        for i in range(5):
            edges.append([i, (i + 1) % 5])
            edges.append([i, 5 + i])
            edges.append([(i + 1) % 5, 5 + ((i + 1) % 5)])
        for u in range(5):
            edges.append([5 + u, 10])
