        def dfs(row, depth):
            nonlocal best, nodes, used_local
            nodes[0] += 1
            if nodes[0] > node_limit:
                return
            if depth == m:
                if row[-1] < best['max']:
                    best['max'] = row[-1]
                    best['row'] = list(row)
                return
            x = row[-1] + 1
            while x < best['max']:
                new = {x - a for a in row}
                if not (new & used_local) and not (new & forbidden):
                    used_local |= new
                    row.append(x)
                    dfs(row, depth + 1)
                    row.pop()
                    used_local -= new
                    if nodes[0] > node_limit:
                        return
                x += 1
