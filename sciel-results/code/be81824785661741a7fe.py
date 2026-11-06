                seq.append(x)
                dfs(seq, sums, new_suffix_sums, x + 1)
                seq.pop()
