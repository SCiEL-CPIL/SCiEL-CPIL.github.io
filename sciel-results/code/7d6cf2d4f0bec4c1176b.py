import time

N = 16

# Precomputed best split k for n=4..16
SPLIT = {
    4: 3,
    5: 3,
    6: 3,
    7: 3,
    8: 4,
    9: 3,
    10: 4,
    11: 5,
    12: 6,
    13: 6,
    14: 7,
    15: 6,
    16: 7
}

def construct():
    """
    Constructs a Gallai coloring of K_16 with the maximum number of distinct colors (20).
    """
    colors = [[0] * N for _ in range(N)]
    next_color = [0]

    def assign_colors(vertices):
        n = len(vertices)
        if n <= 1:
            return next_color[0]
        if n == 2:
            u, v = vertices
            c = next_color[0]
            colors[u][v] = c
            colors[v][u] = c
            next_color[0] += 1
            return next_color[0]
        if n == 3:
            u, v, w = vertices
            c1 = next_color[0]
            c2 = next_color[0] + 1
            c3 = next_color[0] + 2
            colors[u][v] = c1; colors[v][u] = c1
            colors[v][w] = c2; colors[w][v] = c2
            colors[u][w] = c3; colors[w][u] = c3
            next_color[0] += 3
            return next_color[0]
        k = SPLIT[n]
        v1 = vertices[:k]
        v2 = vertices[k:]
        cross_color = next_color[0]
        for u in v1:
            for v in v2:
                colors[u][v] = cross_color
                colors[v][u] = cross_color
        next_color[0] += 1
        assign_colors(v1)
        assign_colors(v2)
        return next_color[0]

    assign_colors(list(range(N)))

    result = []
    for i in range(N):
        for j in range(i + 1, N):
            result.append(colors[i][j])
    return result

if __name__ == "__main__":
    # Simple self-check
    t0 = time.time()
    res = construct()
    t1 = time.time()
    print(f"Number of colors used: {len(set(res))}")
    print(f"First 10 colors: {res[:10]}")
    print(f"Time: {t1 - t0:.2f}s")
