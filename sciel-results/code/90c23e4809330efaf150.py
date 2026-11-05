                    temp = inter
                    while temp:
                        lsb = temp & -temp
                        c = lsb.bit_length() - 1
                        temp ^= lsb
                        
                        swaps.append((u, c))
                        swaps.append((a, c))
                        swaps.append((b, c))
