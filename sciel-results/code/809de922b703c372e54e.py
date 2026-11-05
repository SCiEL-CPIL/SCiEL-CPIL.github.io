            new_cost = 0
            # Improper
            for i in range(num_edges):
                c_i = colors[i]
                for j in edge_neighbors[i]:
                    if j > i and colors[j] == c_i:
                        new_cost += 100
            # Non-rainbow
            for c4 in c4s:
                if len({colors[c4[0]], colors[c4[1]], colors[c4[2]], colors[c4[3]]}) < 4:
                    new_cost += 1
