    def remove_val(val, c_arr, counts_arr):
        counts_arr[val + c_arr] -= 1
        counts_arr[2 * val] -= 1

    def add_val(val, c_arr, counts_arr):
        counts_arr[val + c_arr] += 1
        counts_arr[2 * val] += 1
